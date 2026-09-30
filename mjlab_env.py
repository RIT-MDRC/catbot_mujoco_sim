"""Batched Catbot velocity tracking on mjlab, implementing RSL-RL's VecEnv API."""

from __future__ import annotations

import math
from typing import TypedDict

import mujoco
import torch
import warp as wp
from mjlab.sim import Simulation, SimulationCfg
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from catbot_env import CatbotEnv


class CatbotConfig(TypedDict):
    num_envs: int
    device: str
    seed: int
    max_episode_length: int
    frame_skip: int
    action_smoothing: float
    domain_randomization: bool
    foot_clearance_weight: float
    foot_clearance_target: float
    action_acceleration_weight: float
    air_time_weight: float
    air_time_min: float
    air_time_max: float


class CatbotMjlabEnv(VecEnv):
    """Batched locomotion with observable action and gait history.

    Start with fixed masses/friction. Reset randomizes base position, velocity
    and the velocity command; joint quaternions retain their reference pose.
    All worlds, policy inputs and rewards stay on the selected device.
    """

    def __init__(self, num_envs=256, device="cuda:0", seed=0, max_episode_length=1000):
        if num_envs < 1 or max_episode_length < 1:
            raise ValueError("num_envs and max_episode_length must be positive")
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.cfg: CatbotConfig = {
            "num_envs": num_envs,
            "device": device,
            "seed": seed,
            "max_episode_length": max_episode_length,
            "frame_skip": 10,
            "action_smoothing": 0.8,
            "domain_randomization": False,
            "foot_clearance_weight": 0.2,
            "foot_clearance_target": 0.06,
            "action_acceleration_weight": 0.5,
            "air_time_weight": 0.2,
            "air_time_min": 0.12,
            "air_time_max": 0.35,
        }
        self.generator = torch.Generator(device=device).manual_seed(seed)
        wp.init()
        if device.startswith("cuda"):
            # Graph capture cannot use PyTorch's default CUDA stream. Keep a
            # Warp-owned stream alive and share it with all subsequent policy
            # tensor writes and simulation launches on this thread.
            self._warp_stream = wp.Stream(device)
            self._torch_stream = wp.stream_to_torch(self._warp_stream)
            self._torch_stream.wait_stream(torch.cuda.current_stream(device))
            wp.set_stream(self._warp_stream)
            torch.cuda.set_stream(self._torch_stream)
        model = mujoco.MjModel.from_xml_string(CatbotEnv._render_mjcf())
        self.sim = Simulation(num_envs, SimulationCfg(), model=model, device=device)
        self.num_actions = model.nu
        self.frame_skip = self.cfg["frame_skip"]
        self.dt = self.frame_skip * model.opt.timestep
        self.center = torch.tensor(
            model.actuator_ctrlrange.mean(axis=1), dtype=torch.float32, device=device
        )
        self.half_range = torch.tensor(
            (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]) / 2,
            dtype=torch.float32,
            device=device,
        )
        self.feet = [model.geom(f"{leg}_foot").id for leg in ("fl", "fr", "rl", "rr")]
        self.ground = model.geom("ground").id
        self.foot_radius = torch.tensor(
            model.geom_size[self.feet, 0], dtype=torch.float32, device=device
        )
        self.active_actions = self.half_range > 0
        self.previous_action = torch.zeros(num_envs, model.nu, device=device)
        self.previous_action_delta = torch.zeros_like(self.previous_action)
        self.feet_air_time = torch.zeros(num_envs, 4, device=device)
        self.previous_contact = torch.zeros(
            num_envs, 4, dtype=torch.bool, device=device
        )
        self.has_stance = torch.zeros_like(self.previous_contact)
        self.command = torch.zeros(num_envs, 3, device=device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.previous_feet = torch.zeros(num_envs, 4, 3, device=device)
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, ids):
        if ids.numel() == 0:
            return
        self.sim.reset(ids)
        count = ids.numel()
        self.sim.data.qpos[ids, :3] += self._uniform((count, 3), -0.02, 0.02)
        self.sim.data.qvel[ids] = self._uniform(
            (count, self.sim.mj_model.nv), -0.02, 0.02
        )
        low = torch.tensor([-0.3, -0.1, -0.8], device=self.device)
        span = torch.tensor([1.1, 0.2, 1.6], device=self.device)
        self.command[ids] = low + self._uniform((count, 3), 0, 1) * span
        self.previous_action[ids] = 0
        self.previous_action_delta[ids] = 0
        self.feet_air_time[ids] = 0
        self.episode_length_buf[ids] = 0
        self.sim.data.ctrl[ids] = self.center
        self.sim.forward()
        self.previous_contact[ids] = self.foot_contacts()[ids]
        self.has_stance[ids] = self.previous_contact[ids]
        self.previous_feet[ids] = self.sim.data.geom_xpos[ids][:, self.feet]

    def _uniform(self, shape, low, high):
        return low + (high - low) * torch.rand(
            shape, device=self.device, generator=self.generator
        )

    def get_observations(self):
        data = self.sim.data
        obs = torch.cat(
            (
                data.qpos[:, 2:7],
                data.qvel[:, :6],
                data.qpos[:, 7:],
                data.qvel[:, 6:],
                self.command,
                self.previous_action,
                self.previous_action_delta,
                self.feet_air_time / self.cfg["air_time_max"],
                self.previous_contact.float(),
                self.has_stance.float(),
            ),
            dim=1,
        )
        return TensorDict({"actor": obs, "critic": obs}, batch_size=[self.num_envs])

    def foot_contacts(self):
        """Reduce the shared Warp contact buffer by world and foot, on-device."""
        contacts = self.sim.data.contact
        geom = contacts.geom[:]
        world = contacts.worldid[:].long()
        valid = torch.arange(world.numel(), device=self.device) < self.sim.data.nacon[0]
        valid &= (world >= 0) & (world < self.num_envs)
        touching = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        for index, foot in enumerate(self.feet):
            match = ((geom[:, 0] == foot) & (geom[:, 1] == self.ground)) | (
                (geom[:, 1] == foot) & (geom[:, 0] == self.ground)
            )
            touching[:, index].scatter_add_(
                0, world.clamp(0, self.num_envs - 1), (valid & match).long()
            )
        return touching > 0

    def gait_terms(self, action, contact):
        """Evaluate history-dependent terms without advancing their state."""
        acceleration = (
            (action - self.previous_action - self.previous_action_delta)
            .square()[:, self.active_actions]
            .mean(dim=1)
        )
        # A foot must first establish stance; the initial reset fall is not a step.
        touchdown = contact & self.has_stance & (self.feet_air_time > 0)
        duration = (
            (self.feet_air_time - self.cfg["air_time_min"])
            / (self.cfg["air_time_max"] - self.cfg["air_time_min"])
        ).clamp(0, 1)
        moving = (self.command[:, :2].norm(dim=1) > 0.1) | (
            self.command[:, 2].abs() > 0.1
        )
        air_time = (duration * touchdown).mean(dim=1) * moving
        return acceleration, air_time

    def reward_terms(self, action, contact=None):
        data = self.sim.data
        quat = data.qpos[:, 3:7]
        up = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        tracking = torch.exp(
            -2 * (data.qvel[:, [0, 1, 5]] - self.command).square().sum(dim=1)
        )
        upright = up.clamp(min=0)
        height = torch.exp(-40 * (data.qpos[:, 2] - 0.32).square())
        control = 0.015 * action[:, self.active_actions].square().mean(dim=1)
        rate = 0.1 * (action - self.previous_action)[
            :, self.active_actions
        ].square().mean(dim=1)
        feet = data.geom_xpos[:, self.feet]
        velocity = (feet - self.previous_feet) / self.dt
        if contact is None:
            contact = self.foot_contacts()
        acceleration, air_time = self.gait_terms(action, contact)
        # Exclude touchdown displacement from the stance-slip estimate.
        stance = contact & self.previous_contact
        drag = (velocity[:, :, :2].square().sum(dim=2) * stance).sum(
            dim=1
        ) / stance.sum(dim=1).clamp(min=1)
        swing = ~contact
        # Spherical feet over the scene's horizontal ground plane. Normalize
        # the deficit so the coefficient has a meaningful reward-scale value.
        foot_height = (
            feet[:, :, 2] - self.foot_radius - data.geom_xpos[:, self.ground, 2:3]
        )
        deficit = (1 - foot_height / self.cfg["foot_clearance_target"]).clamp(0, 1)
        clearance = (deficit.square() * swing).sum(dim=1) / swing.sum(dim=1).clamp(
            min=1
        )
        reward = (
            0.2
            + tracking
            + 0.5 * upright
            + 0.3 * height
            - control
            - rate
            - 0.05 * drag
            - self.cfg["foot_clearance_weight"] * clearance
            - self.cfg["action_acceleration_weight"] * acceleration
            + self.cfg["air_time_weight"] * air_time
        )
        return (
            reward,
            up,
            {
                "/reward": reward.mean(),
                "/tracking": tracking.mean(),
                "/upright": upright.mean(),
                "/foot_drag": drag.mean(),
                "/foot_clearance": clearance.mean(),
                "/foot_clearance_penalty": (
                    self.cfg["foot_clearance_weight"] * clearance.mean()
                ),
                "/action_acceleration_penalty": (
                    self.cfg["action_acceleration_weight"] * acceleration.mean()
                ),
                "/air_time_reward": self.cfg["air_time_weight"] * air_time.mean(),
            },
        )

    def step(self, actions):
        if actions.shape != self.previous_action.shape:
            raise ValueError(f"Expected actions of shape {self.previous_action.shape}")
        smoothing = self.cfg["action_smoothing"]
        action = smoothing * self.previous_action + (1 - smoothing) * actions.clamp(
            -1, 1
        )
        action = action * self.active_actions
        self.sim.data.ctrl[:] = self.center + action * self.half_range
        for _ in range(self.frame_skip):
            self.sim.step()
        self.sim.forward()
        self.episode_length_buf += 1
        contact = self.foot_contacts()
        reward, up, metrics = self.reward_terms(action, contact)
        finite = torch.isfinite(self.sim.data.qpos).all(dim=1) & torch.isfinite(
            self.sim.data.qvel
        ).all(dim=1)
        finite &= torch.isfinite(reward)
        failed = (
            ~finite
            | (self.sim.data.qpos[:, 2] < 0.14)
            | (up < math.cos(math.radians(20)))
        )
        timeouts = (self.episode_length_buf >= self.max_episode_length) & ~failed
        done = failed | timeouts
        reward = torch.where(failed, torch.full_like(reward, -1), reward)
        metrics["/reward"] = reward.mean()
        self.previous_action_delta.copy_(action - self.previous_action)
        self.previous_action.copy_(action)
        self.feet_air_time.add_(self.dt).clamp_(max=self.cfg["air_time_max"])
        self.feet_air_time.masked_fill_(contact, 0)
        self.has_stance |= contact
        self.previous_contact.copy_(contact)
        self.previous_feet.copy_(self.sim.data.geom_xpos[:, self.feet])
        self.reset(done.nonzero(as_tuple=False).flatten())
        metrics = {key: torch.nan_to_num(value) for key, value in metrics.items()}
        metrics["/failures"] = failed.float().mean()
        return (
            self.get_observations(),
            reward,
            done,
            {"time_outs": timeouts, "log": metrics},
        )
