"""Batched Catbot velocity tracking on mjlab, implementing RSL-RL's VecEnv API."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict, cast, get_type_hints

import mujoco
import torch
import warp as wp
import yaml
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
    tracking_weight: float
    tilt_weight: float
    tilt_reference_degrees: float
    tilt_rate_weight: float
    upright_weight: float
    height_weight: float
    motion_budget: float
    motion_budget_weight: float
    control_weight: float
    action_rate_weight: float
    stance_radius: float
    stance_boundary_weight: float
    foot_drag_weight: float
    foot_clearance_weight: float
    foot_clearance_target: float
    action_acceleration_weight: float
    air_time_weight: float
    air_time_min: float
    air_time_max: float
    air_time_limit: float
    min_support_feet: int
    support_deficit_weight: float
    gait_contact_weight: float
    trot_speed: float
    bound_speed: float
    gait_transition_seconds: float
    command_interval: int
    prolonged_swing_weight: float


DEFAULT_CONFIG = Path(__file__).resolve().parent / "assets" / "catbot_config.yaml"


def load_catbot_config(source=None) -> CatbotConfig:
    """Read YAML defaults, a custom YAML file, or a saved run's resolved config."""
    if isinstance(source, dict):
        config = dict(source)
    else:
        path = DEFAULT_CONFIG if source is None else Path(source)
        config = yaml.safe_load(path.read_text())
    schema = get_type_hints(CatbotConfig)
    if not isinstance(config, dict) or config.keys() != schema.keys():
        raise ValueError("Catbot config must contain exactly the CatbotConfig settings")
    for key, expected in schema.items():
        value = config[key]
        valid = type(value) is expected
        if expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"Invalid {key}: expected {expected.__name__}")
    for key in ("num_envs", "max_episode_length", "frame_skip", "command_interval"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key, value in config.items():
        if key.endswith("_weight") and value < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not 0 <= config["action_smoothing"] < 1:
        raise ValueError("action_smoothing must be in [0, 1)")
    if (
        not 0
        <= config["air_time_min"]
        < config["air_time_max"]
        < config["air_time_limit"]
    ):
        raise ValueError("Require 0 <= air_time_min < air_time_max < air_time_limit")
    if not 0 <= config["trot_speed"] < config["bound_speed"]:
        raise ValueError("Require 0 <= trot_speed < bound_speed")
    if config["foot_clearance_target"] <= 0 or config["gait_transition_seconds"] <= 0:
        raise ValueError(
            "Clearance target and gait transition duration must be positive"
        )
    if not 0 <= config["min_support_feet"] <= 4:
        raise ValueError("min_support_feet must be between zero and four")
    if not 0 < config["tilt_reference_degrees"] < 90:
        raise ValueError("tilt_reference_degrees must be between zero and 90")
    if config["motion_budget"] <= 0:
        raise ValueError("motion_budget must be positive")
    if config["stance_radius"] <= 0:
        raise ValueError("stance_radius must be positive")
    if config["domain_randomization"]:
        raise ValueError("Domain randomization is not implemented for mjlab")
    return cast(CatbotConfig, config)


class CatbotMjlabEnv(VecEnv):
    """Batched locomotion with observable action and gait history.

    Start with fixed masses/friction. Reset randomizes base position, velocity
    and the velocity command; joint quaternions retain their reference pose.
    All worlds, policy inputs and rewards stay on the selected device.
    """

    cfg: CatbotConfig

    def __init__(
        self,
        num_envs=None,
        device=None,
        seed=None,
        max_episode_length=None,
        *,
        config=None,
    ):
        self.cfg = load_catbot_config(config)
        for key, value in {
            "num_envs": num_envs,
            "device": device,
            "seed": seed,
            "max_episode_length": max_episode_length,
        }.items():
            if value is not None:
                self.cfg[key] = value
        num_envs = self.cfg["num_envs"]
        device = self.cfg["device"]
        seed = self.cfg["seed"]
        max_episode_length = self.cfg["max_episode_length"]
        if num_envs < 1 or max_episode_length < 1:
            raise ValueError("num_envs and max_episode_length must be positive")
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
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
        # Hip body origins coincide with the ball-joint anchors in this model.
        self.hips = [model.body(f"{leg}_hip").id for leg in ("fl", "fr", "rl", "rr")]
        self.ground = model.geom("ground").id
        self.foot_radius = torch.tensor(
            model.geom_size[self.feet, 0], dtype=torch.float32, device=device
        )
        self.active_actions = self.half_range > 0
        # Per-leg hip X, hip Y, knee angular velocities; exclude locked hip Z.
        motion_dofs, motion_ranges = [], []
        for leg in ("fl", "fr", "rl", "rr"):
            hip = model.joint(f"{leg}_hip_ball").id
            knee = model.joint(f"{leg}_knee").id
            motion_dofs.extend(
                [
                    int(model.jnt_dofadr[hip]),
                    int(model.jnt_dofadr[hip]) + 1,
                    int(model.jnt_dofadr[knee]),
                ]
            )
            for name in (f"{leg}_hip_x", f"{leg}_hip_y", f"{leg}_knee_actuator"):
                limits = model.actuator_ctrlrange[model.actuator(name).id]
                motion_ranges.append(float(limits[1] - limits[0]))
        self.motion_dofs = motion_dofs
        self.motion_ranges = torch.tensor(motion_ranges, device=device)
        self.motion_travel = torch.zeros(num_envs, 12, device=device)
        self.motion_progress = torch.zeros(num_envs, device=device)
        self.motion_cost = torch.zeros(num_envs, device=device)
        self.previous_action = torch.zeros(num_envs, model.nu, device=device)
        self.previous_action_delta = torch.zeros_like(self.previous_action)
        self.feet_air_time = torch.zeros(num_envs, 4, device=device)
        self.previous_contact = torch.zeros(
            num_envs, 4, dtype=torch.bool, device=device
        )
        self.has_stance = torch.zeros_like(self.previous_contact)
        self.gait_phase = torch.zeros(num_envs, device=device)
        self.gait_blend = torch.zeros(num_envs, device=device)
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
        self.sample_commands(ids)
        self.gait_phase[ids] = self._uniform((count,), 0, 1)
        self.gait_blend[ids] = self.target_gait_blend()[ids]
        self.motion_travel[ids] = 0
        self.motion_progress[ids] = 0
        self.motion_cost[ids] = 0
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

    def sample_commands(self, ids):
        low = torch.tensor([-0.8, -0.1, -0.8], device=self.device)
        span = torch.tensor([1.6, 0.2, 1.6], device=self.device)
        self.command[ids] = low + self._uniform((ids.numel(), 3), 0, 1) * span

    def moving_command(self):
        return (self.command[:, :2].norm(dim=1) > 0.1) | (
            self.command[:, 2].abs() > 0.1
        )

    def target_gait_blend(self):
        return (
            (self.command[:, 0].abs() - self.cfg["trot_speed"])
            / (self.cfg["bound_speed"] - self.cfg["trot_speed"])
        ).clamp(0, 1)

    def advance_gait(self):
        limit = self.dt / self.cfg["gait_transition_seconds"]
        self.gait_blend.add_(
            (self.target_gait_blend() - self.gait_blend).clamp(-limit, limit)
        )
        # 1.5 Hz trot -> 2 Hz bound; freeze the clock while standing.
        self.gait_phase.add_(
            self.dt * (1.5 + 0.5 * self.gait_blend) * self.moving_command()
        ).remainder_(1)

    def gait_contact_target(self):
        # FL, FR, RL, RR. Keep FL/RL and FR/RR half a cycle apart throughout
        # the transition, avoiding an intermediate all-feet-swing target.
        b = self.gait_blend
        offsets = torch.stack(
            (torch.zeros_like(b), 0.5 * (1 - b), torch.full_like(b, 0.5), 1 - 0.5 * b),
            dim=1,
        )
        angle = 2 * math.pi * (self.gait_phase[:, None] + offsets)
        # 60% stance with soft boundaries and overlap between supporting pairs.
        target = torch.sigmoid(12 * (torch.cos(angle) - math.cos(math.pi * 0.6)))
        return torch.where(
            self.moving_command()[:, None], target, torch.ones_like(target)
        )

    def heading_linear_velocity(self):
        """World linear velocity expressed in the body's horizontal heading frame.

        Roll/pitch do not turn vertical bouncing into commanded forward motion.
        MuJoCo free-joint angular velocities are already body-local.
        """
        w, x, y, z = self.sim.data.qpos[:, 3:7].unbind(dim=1)
        forward_x = 1 - 2 * (y.square() + z.square())
        forward_y = 2 * (x * y + w * z)
        norm = (forward_x.square() + forward_y.square()).sqrt().clamp(min=1e-6)
        c, s = forward_x / norm, forward_y / norm
        vx, vy, vz = self.sim.data.qvel[:, :3].unbind(dim=1)
        return torch.stack((c * vx + s * vy, -s * vx + c * vy, vz), dim=1)

    def get_observations(self):
        data = self.sim.data
        obs = torch.cat(
            (
                data.qpos[:, 2:7],
                self.heading_linear_velocity(),
                data.qvel[:, 3:6],
                data.qpos[:, 7:],
                data.qvel[:, 6:],
                self.command,
                self.previous_action,
                self.previous_action_delta,
                self.feet_air_time / self.cfg["air_time_limit"],
                self.previous_contact.float(),
                self.has_stance.float(),
                torch.sin(2 * math.pi * self.gait_phase)[:, None],
                torch.cos(2 * math.pi * self.gait_phase)[:, None],
                self.gait_blend[:, None],
                self.motion_travel,
                self.motion_progress[:, None],
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
        duration *= 1 - self.swing_overdue(self.feet_air_time)
        moving = self.moving_command()
        air_time = (duration * touchdown).mean(dim=1) * moving
        return acceleration, air_time

    def swing_overdue(self, air_time):
        """Fraction of the allowed overrun, per foot, bounded for stability."""
        return (
            (air_time - self.cfg["air_time_max"])
            / (self.cfg["air_time_limit"] - self.cfg["air_time_max"])
        ).clamp(0, 1)

    def reward_terms(self, action, contact=None):
        data = self.sim.data
        quat = data.qpos[:, 3:7]
        up = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        commanded_velocity = torch.cat(
            (self.heading_linear_velocity()[:, :2], data.qvel[:, 5:6]), dim=1
        )
        tracking = torch.exp(
            -2 * (commanded_velocity - self.command).square().sum(dim=1)
        )
        tilt = (1 - up).clamp(min=0) / (
            1 - math.cos(math.radians(self.cfg["tilt_reference_degrees"]))
        )
        tilt_rate = data.qvel[:, 3:5].square().sum(dim=1)
        upright = up.clamp(min=0)
        height = torch.exp(-40 * (data.qpos[:, 2] - 0.32).square())
        control = action[:, self.active_actions].square().mean(dim=1)
        rate = (
            (action - self.previous_action)[:, self.active_actions].square().mean(dim=1)
        )
        feet = data.geom_xpos[:, self.feet]
        velocity = (feet - self.previous_feet) / self.dt
        if contact is None:
            contact = self.foot_contacts()
        # On the horizontal ground plane, projecting the hip vertically leaves
        # its XY coordinates unchanged. Foot spheres contact below their centers.
        hip_xy = data.xpos[:, self.hips, :2]
        stance_distance = (feet[:, :, :2] - hip_xy).norm(dim=2)
        stance_excess = (stance_distance / self.cfg["stance_radius"] - 1).clamp(min=0)
        # Fixed denominator: another foot cannot dilute an existing violation.
        stance_boundary = (stance_excess.square() * contact).mean(dim=1)
        acceleration, air_time = self.gait_terms(action, contact)
        gait_match = 1 - (contact.float() - self.gait_contact_target()).abs().mean(
            dim=1
        )
        support_deficit = (self.cfg["min_support_feet"] - contact.sum(dim=1)).clamp(
            min=0
        )
        # Exclude touchdown displacement from the stance-slip estimate.
        stance = contact & self.previous_contact
        drag = (velocity[:, :, :2].square().sum(dim=2) * stance).sum(
            dim=1
        ) / stance.sum(dim=1).clamp(min=1)
        swing = ~contact
        prolonged_swing = (
            self.swing_overdue(self.feet_air_time + self.dt) * swing
        ).sum(dim=1)
        # Spherical feet over the scene's horizontal ground plane. Normalize
        # the deficit so the coefficient has a meaningful reward-scale value.
        foot_height = (
            feet[:, :, 2] - self.foot_radius - data.geom_xpos[:, self.ground, 2:3]
        )
        deficit = (1 - foot_height / self.cfg["foot_clearance_target"]).clamp(0, 1)
        clearance = (deficit.square() * swing).mean(dim=1)
        reward = (
            0.2
            + self.cfg["gait_contact_weight"] * gait_match
            + self.cfg["tracking_weight"] * tracking
            + self.cfg["upright_weight"] * upright
            - self.cfg["tilt_weight"] * tilt
            - self.cfg["tilt_rate_weight"] * tilt_rate
            + self.cfg["height_weight"] * height
            - self.cfg["motion_budget_weight"] * self.motion_cost
            - self.cfg["control_weight"] * control
            - self.cfg["action_rate_weight"] * rate
            - self.cfg["foot_drag_weight"] * drag
            - self.cfg["stance_boundary_weight"] * stance_boundary
            - self.cfg["support_deficit_weight"] * support_deficit
            - self.cfg["prolonged_swing_weight"] * prolonged_swing
            - self.cfg["foot_clearance_weight"] * clearance
            - self.cfg["action_acceleration_weight"] * acceleration
            + self.cfg["air_time_weight"] * air_time
        )
        return (
            reward,
            up,
            {
                "/reward": reward.mean(),
                "/motion_budget_penalty": self.cfg["motion_budget_weight"]
                * self.motion_cost.mean(),
                "/hip_travel": self.motion_travel.reshape(self.num_envs, 4, 3)[
                    :, :, :2
                ].mean(),
                "/knee_travel": self.motion_travel.reshape(self.num_envs, 4, 3)[
                    :, :, 2
                ].mean(),
                "/gait_contact_reward": self.cfg["gait_contact_weight"]
                * gait_match.mean(),
                "/bound_blend": self.gait_blend.mean(),
                "/support_deficit_penalty": self.cfg["support_deficit_weight"]
                * support_deficit.float().mean(),
                "/tracking": tracking.mean(),
                "/upright": upright.mean(),
                "/tilt_penalty": self.cfg["tilt_weight"] * tilt.mean(),
                "/tilt_rate_penalty": self.cfg["tilt_rate_weight"] * tilt_rate.mean(),
                "/foot_drag": drag.mean(),
                "/stance_boundary_penalty": self.cfg["stance_boundary_weight"]
                * stance_boundary.mean(),
                "/foot_clearance": clearance.mean(),
                "/foot_clearance_penalty": (
                    self.cfg["foot_clearance_weight"] * clearance.mean()
                ),
                "/action_acceleration_penalty": (
                    self.cfg["action_acceleration_weight"] * acceleration.mean()
                ),
                "/prolonged_swing_penalty": self.cfg["prolonged_swing_weight"]
                * prolonged_swing.mean(),
                "/air_time_reward": self.cfg["air_time_weight"] * air_time.mean(),
            },
        )

    def accumulate_motion(self, travel):
        """Charge only additional squared budget excess, not time spent above it.

        Travel is absolute joint-axis angular motion / usable actuator range.
        Adding motion in another joint cannot cancel a joint's excess.
        """
        before = (self.motion_travel - self.cfg["motion_budget"]).clamp(min=0).square()
        self.motion_travel.add_(travel)
        after = (self.motion_travel - self.cfg["motion_budget"]).clamp(min=0).square()
        self.motion_cost.add_((after - before).mean(dim=1))

    def step(self, actions):
        if actions.shape != self.previous_action.shape:
            raise ValueError(f"Expected actions of shape {self.previous_action.shape}")
        smoothing = self.cfg["action_smoothing"]
        action = smoothing * self.previous_action + (1 - smoothing) * actions.clamp(
            -1, 1
        )
        action = action * self.active_actions
        self.sim.data.ctrl[:] = self.center + action * self.half_range
        self.motion_cost.zero_()
        for _ in range(self.frame_skip):
            self.sim.step()
            travel = (
                self.sim.data.qvel[:, self.motion_dofs].abs()
                * self.sim.mj_model.opt.timestep
            )
            self.accumulate_motion(travel / self.motion_ranges)
        self.sim.forward()
        self.episode_length_buf += 1
        self.advance_gait()
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
        # A full cadence-length window, even while standing (the gait clock
        # itself freezes at rest). Check/reset after charging the final step.
        self.motion_progress.add_(self.dt * (1.5 + 0.5 * self.gait_blend))
        completed = self.motion_progress >= 1
        self.motion_travel[completed] = 0
        self.motion_progress[completed] = 0
        self.previous_action_delta.copy_(action - self.previous_action)
        self.previous_action.copy_(action)
        self.feet_air_time.add_(self.dt).clamp_(max=self.cfg["air_time_limit"])
        self.feet_air_time.masked_fill_(contact, 0)
        self.has_stance |= contact
        self.previous_contact.copy_(contact)
        self.previous_feet.copy_(self.sim.data.geom_xpos[:, self.feet])
        resample = (self.episode_length_buf % self.cfg["command_interval"] == 0) & ~done
        self.sample_commands(resample.nonzero(as_tuple=False).flatten())
        self.reset(done.nonzero(as_tuple=False).flatten())
        metrics = {key: torch.nan_to_num(value) for key, value in metrics.items()}
        metrics["/failures"] = failed.float().mean()
        return (
            self.get_observations(),
            reward,
            done,
            {"time_outs": timeouts, "log": metrics},
        )
