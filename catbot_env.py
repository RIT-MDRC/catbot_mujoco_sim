"""Gymnasium environment for training locomotion policies on the Catbot model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from jinja2 import Environment, FileSystemLoader
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ObservationArray = NDArray[np.float32]
ActionArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]


class CatbotEnv(gym.Env[ObservationArray, ActionArray]):
    """Velocity-tracking quadruped environment with normalized position actions.

    Actions are 16 values in ``[-1, 1]``. They are rescaled to the MuJoCo
    position-actuator ranges: hip x/y targets and one knee target for each
    limb. The hip-z slots remain fixed at zero for real-robot compatibility.
    The observation contains only state available to a robot controller plus
    the velocity command and last action.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    _renderer: mujoco.Renderer | None = None
    _viewer: Any | None = None

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 1_000,
        domain_randomization: bool = True,
        action_smoothing: float = 0.8,
        action_smoothing_enabled: bool = True,
    ) -> None:
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        if not 0.0 <= action_smoothing < 1.0:
            raise ValueError("action_smoothing must be in the range [0.0, 1.0)")
        self.action_smoothing = action_smoothing
        self.action_smoothing_enabled = action_smoothing_enabled
        self.model = mujoco.MjModel.from_xml_string(self._render_mjcf())
        self.data = mujoco.MjData(self.model)
        self._step_count = 0
        self._previous_action: FloatArray = np.zeros(self.model.nu, dtype=np.float64)
        self._command: FloatArray = np.zeros(3, dtype=np.float64)
        self._initial_qpos: FloatArray = self.model.qpos0.copy()
        self._initial_body_mass: FloatArray = self.model.body_mass.copy()
        self._initial_ground_friction: FloatArray = self.model.geom_friction[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        ].copy()
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "quadruped"
        )
        self._ground_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self._foot_ids: NDArray[np.int32] = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("fl_foot", "fr_foot", "rl_foot", "rr_foot")
            ],
            dtype=np.int32,
        )
        self._control_timestep = self.model.opt.timestep * self.frame_skip
        self._previous_foot_positions: FloatArray = np.zeros(
            (len(self._foot_ids), 3), dtype=np.float64
        )
        self._ctrl_center: FloatArray = self.model.actuator_ctrlrange.mean(axis=1)
        self._ctrl_half_range: FloatArray = (
            np.diff(self.model.actuator_ctrlrange, axis=1).ravel() / 2
        )

        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
        )
        observation: ObservationArray = self._get_observation()
        self.observation_space = gym.spaces.Box(
            -100.0, 100.0, shape=observation.shape, dtype=np.float32
        )

    @staticmethod
    def _render_mjcf() -> str:
        asset_dir = Path(__file__).parent / "assets"
        templates = Environment(loader=FileSystemLoader(asset_dir))
        robot = templates.get_template("robot.xml.j2").render()
        return templates.get_template("world.xml.j2").render(
            worldbody=robot, mujoco_config=""
        )

    def _get_observation(self) -> ObservationArray:
        # Base height/quaternion/velocity; all joint pose and velocity; command and action.
        return np.concatenate(
            (
                self.data.qpos[2:7],
                self.data.qvel[:6],
                self.data.qpos[7:],
                self.data.qvel[6:],
                self._command,
                self._previous_action,
            )
        ).astype(np.float32, copy=False)

    def _apply_domain_randomization(self) -> None:
        self.model.body_mass[:] = self._initial_body_mass
        self.model.geom_friction[self._ground_id] = self._initial_ground_friction
        if not self.domain_randomization:
            return
        self.model.body_mass[self._torso_id] *= self.np_random.uniform(0.9, 1.1)
        self.model.geom_friction[self._ground_id, 0] *= self.np_random.uniform(0.7, 1.3)

    def _smooth_action(self, action: FloatArray) -> FloatArray:
        """Apply an exponential moving average to a clipped policy action."""
        if not self.action_smoothing_enabled:
            return action.copy()
        return (
            self.action_smoothing * self._previous_action
            + (1.0 - self.action_smoothing) * action
        )

    def _foot_contacts_ground(self, foot_id: int) -> bool:
        """Return whether a foot geom is currently touching the ground geom."""
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if {contact.geom1, contact.geom2} == {foot_id, self._ground_id}:
                return True
        return False

    def _compute_foot_penalties(self) -> tuple[float, float]:
        """Compute ground-drag and low swing-foot penalties."""
        foot_positions = self.data.geom_xpos[self._foot_ids].copy()
        foot_velocities = (
            foot_positions - self._previous_foot_positions
        ) / self._control_timestep
        drag_costs: list[float] = []
        clearance_costs: list[float] = []
        for foot_id, foot_position, foot_velocity in zip(
            self._foot_ids, foot_positions, foot_velocities
        ):
            touching_ground = self._foot_contacts_ground(int(foot_id))
            if touching_ground:
                drag_costs.append(float(np.square(foot_velocity[:2]).sum()))
            else:
                clearance_error = max(0.0, 0.06 - float(foot_position[2]))
                clearance_costs.append(clearance_error**2)

        drag_cost = float(np.mean(drag_costs)) if drag_costs else 0.0
        clearance_cost = float(np.mean(clearance_costs)) if clearance_costs else 0.0
        return drag_cost, clearance_cost

    def _compute_reward(
        self, action: np.ndarray, torso_up: float, base_height: float
    ) -> tuple[float, dict[str, float]]:
        """Compute the reward and its component diagnostics for the current state."""
        # Free-joint velocity layout is linear XYZ followed by angular XYZ.
        # The command is forward, lateral, and yaw, so yaw uses angular Z.
        actual_velocity = np.array(
            [self.data.qvel[0], self.data.qvel[1], self.data.qvel[5]],
            dtype=np.float64,
        )
        velocity_error = actual_velocity - self._command
        # Reward for tracking the commanded velocity, using a Gaussian function of the error.
        tracking_reward = float(np.exp(-2.0 * np.square(velocity_error).sum()))

        # Reward for keeping the torso upright, based on the z-axis of the torso's rotation matrix.
        upright_reward = max(torso_up, 0.0)

        # Reward for keeping the base height near the target height of 0.4 m.
        target_height = 0.40
        height_error = base_height - target_height
        height_reward = float(np.exp(-40.0 * height_error**2))

        # Reduce large control inputs to encourage energy efficiency and smoother motion.
        control_cost = 0.015 * float(np.square(action).mean())

        # Penalize rapid changes between consecutive actions to discourage jitter.
        action_rate_cost = 0.1 * float(np.square(action - self._previous_action).mean())
        foot_drag_cost, foot_clearance_cost = self._compute_foot_penalties()

        reward = (
            0.2
            + tracking_reward
            + 0.5 * upright_reward
            + 0.3 * height_reward
            - control_cost
            - action_rate_cost
            - 0.05 * foot_drag_cost
            - 0.02 * foot_clearance_cost
        )
        return reward, {
            "tracking_reward": tracking_reward,
            "upright_reward": upright_reward,
            "action_rate_cost": action_rate_cost,
            "foot_drag_cost": foot_drag_cost,
            "foot_clearance_cost": foot_clearance_cost,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationArray, dict[str, Any]]:
        super().reset(seed=seed)
        self._apply_domain_randomization()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qpos[:3] += self.np_random.uniform(-0.02, 0.02, size=3)
        # Integrating through MuJoCo preserves valid ball-joint quaternions.
        joint_perturbation: FloatArray = np.zeros(self.model.nv, dtype=np.float64)
        joint_perturbation[6:] = self.np_random.uniform(
            -0.04, 0.04, size=self.model.nv - 6
        )
        mujoco.mj_integratePos(self.model, self.data.qpos, joint_perturbation, 1.0)
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        command: FloatArray = np.asarray(
            options.get("command")
            if options and "command" in options
            else self.np_random.uniform(low=(-0.3, -0.1, -0.8), high=(0.8, 0.1, 0.8)),
            dtype=np.float64,
        )
        if command.shape != (3,):
            raise ValueError(
                "options['command'] must contain forward, lateral, and yaw velocities"
            )
        self._command = command
        self._previous_action.fill(0.0)
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        self._previous_foot_positions = self.data.geom_xpos[self._foot_ids].copy()
        return self._get_observation(), {"command": self._command.copy()}

    def step(
        self, action: ActionArray
    ) -> tuple[ObservationArray, float, bool, bool, dict[str, Any]]:
        action_fa: FloatArray = np.asarray(action, dtype=np.float64)
        if action_fa.shape != (self.model.nu,):
            raise ValueError(
                f"Expected action shape {(self.model.nu,)}, received {action_fa.shape}"
            )
        action_fa = np.clip(action_fa, -1.0, 1.0)
        action_fa = self._smooth_action(action_fa)
        self.data.ctrl[:] = self._ctrl_center + action_fa * self._ctrl_half_range
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        torso_up = float(self.data.xmat[self._torso_id].reshape(3, 3)[2, 2])
        base_height = float(self.data.qpos[2])
        reward, reward_info = self._compute_reward(action_fa, torso_up, base_height)
        self._previous_action = action_fa.copy()
        self._previous_foot_positions = self.data.geom_xpos[self._foot_ids].copy()

        invalid_state = (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        )
        terminated = (
            invalid_state or base_height < 0.14 or torso_up < np.cos(np.deg2rad(20.0))
        )
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "command": self._command.copy(),
            "base_height": base_height,
            "torso_up": torso_up,
            **reward_info,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> ImageArray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
