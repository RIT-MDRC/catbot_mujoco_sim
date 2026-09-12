"""Open a managed MuJoCo viewer and play one PPO policy episode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from catbot_env import CatbotEnv


class PolicyController:
    """Supplies PPO actions to MuJoCo's managed simulation loop."""

    def __init__(self, environment: CatbotEnv, policy: Any) -> None:
        self.environment = environment
        self.policy = policy
        self.control_period = environment.frame_skip * environment.model.opt.timestep
        self.next_control_time = environment.data.time
        self.step_count = 0
        self.finished = False

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.finished:
            # Hold the final pose still so it can be inspected in the viewer.
            data.qvel[:] = 0
            data.ctrl[:] = 0
            return
        if data.time < self.next_control_time:
            return

        action, _ = self.policy.predict(
            self.environment._get_observation(), deterministic=True
        )
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action = self.environment._smooth_action(action)
        data.ctrl[:] = self.environment._ctrl_center + action * self.environment._ctrl_half_range
        self.environment._previous_action = action
        self.next_control_time = data.time + self.control_period
        self.step_count += 1

        torso_up = float(data.xmat[self.environment._torso_id].reshape(3, 3)[2, 2])
        invalid_state = not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
        self.finished = (
            invalid_state
            or float(data.qpos[2]) < 0.14
            or torso_up < 0.0
            or self.step_count >= self.environment.max_episode_steps
        )
        if self.finished:
            print(
                f"Episode finished after {self.step_count} control steps. "
                "Close the viewer when you are done inspecting it."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Play one Catbot policy episode.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("runs/catbot_ppo.zip"),
        help="Checkpoint to play (default: runs/catbot_ppo.zip).",
    )
    parser.add_argument(
        "--action-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable action smoothing during rollout (default: enabled).",
    )
    args = parser.parse_args()
    if not (
        args.checkpoint.exists() or args.checkpoint.with_suffix(".zip").exists()
    ):
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")

    from mujoco import viewer
    from stable_baselines3 import PPO

    environment = CatbotEnv(
        domain_randomization=False,
        action_smoothing_enabled=args.action_smoothing,
    )
    environment.reset(seed=42, options={"command": [0.4, 0, 0]})
    controller = PolicyController(environment, PPO.load(args.checkpoint, device="cpu"))
    previous_callback = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(controller)
    print("Opening viewer with PPO control. Press Space if the simulation is paused.")
    try:
        viewer.launch(environment.model, environment.data)
    finally:
        mujoco.set_mjcb_control(previous_callback)
        environment.close()


if __name__ == "__main__":
    main()
