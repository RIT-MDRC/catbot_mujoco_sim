"""Batched Catbot velocity tracking on mjlab, implementing RSL-RL's VecEnv API."""

from __future__ import annotations

import math

import mujoco
import torch
import warp as wp
from mjlab.sim import Simulation, SimulationCfg
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from catbot_env import CatbotEnv


class CatbotMjlabEnv(VecEnv):
    """Retain the existing MJCF, action layout, observation and reward terms.

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
        self.cfg = {
            "num_envs": num_envs,
            "device": device,
            "seed": seed,
            "max_episode_length": max_episode_length,
            "frame_skip": 10,
            "action_smoothing": 0.8,
            "domain_randomization": False,
        }
        self.generator = torch.Generator(device=device).manual_seed(seed)
        wp.init()
        if device.startswith("cuda"):
            # Keep policy tensor writes and Warp graph launches on the same stream.
            wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream(device)))
        model = mujoco.MjModel.from_xml_string(CatbotEnv._render_mjcf())
        self.sim = Simulation(num_envs, SimulationCfg(), model=model, device=device)
        self.num_actions = model.nu
        self.frame_skip = 10
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
        self.previous_action = torch.zeros(num_envs, model.nu, device=device)
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
        self.episode_length_buf[ids] = 0
        self.sim.data.ctrl[ids] = self.center
        self.sim.forward()
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

    def reward_terms(self, action):
        data = self.sim.data
        quat = data.qpos[:, 3:7]
        up = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        tracking = torch.exp(
            -2 * (data.qvel[:, [0, 1, 5]] - self.command).square().sum(dim=1)
        )
        upright = up.clamp(min=0)
        height = torch.exp(-40 * (data.qpos[:, 2] - 0.4).square())
        control = 0.015 * action.square().mean(dim=1)
        rate = 0.1 * (action - self.previous_action).square().mean(dim=1)
        feet = data.geom_xpos[:, self.feet]
        velocity = (feet - self.previous_feet) / self.dt
        contact = self.foot_contacts()
        drag = (velocity[:, :, :2].square().sum(dim=2) * contact).sum(
            dim=1
        ) / contact.sum(dim=1).clamp(min=1)
        swing = ~contact
        clearance = ((0.06 - feet[:, :, 2]).clamp(min=0).square() * swing).sum(
            dim=1
        ) / swing.sum(dim=1).clamp(min=1)
        reward = (
            0.2
            + tracking
            + 0.5 * upright
            + 0.3 * height
            - control
            - rate
            - 0.05 * drag
            - 0.02 * clearance
        )
        return (
            reward,
            up,
            {
                "/tracking": tracking.mean(),
                "/upright": upright.mean(),
                "/foot_drag": drag.mean(),
                "/foot_clearance": clearance.mean(),
            },
        )

    def step(self, actions):
        if actions.shape != self.previous_action.shape:
            raise ValueError(f"Expected actions of shape {self.previous_action.shape}")
        action = 0.8 * self.previous_action + 0.2 * actions.clamp(-1, 1)
        self.sim.data.ctrl[:] = self.center + action * self.half_range
        for _ in range(self.frame_skip):
            self.sim.step()
        self.sim.forward()
        self.episode_length_buf += 1
        reward, up, metrics = self.reward_terms(action)
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
        reward = torch.where(finite, reward, torch.full_like(reward, -1))
        self.previous_action.copy_(action)
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
