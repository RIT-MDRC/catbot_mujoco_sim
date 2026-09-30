"""Catbot CPU rollout with optional RSL-RL checkpoint playback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.sim import Simulation, SimulationCfg

from catbot_env import CatbotEnv


def make_simulation() -> Simulation:
    """Reuse the complete existing MJCF, including ball joints and actuators."""
    model = mujoco.MjModel.from_xml_string(CatbotEnv._render_mjcf())
    simulation = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device="cpu")
    simulation.reset()
    simulation.forward()
    # As in the manual viewer, hold the initial joint pose without a startup kick.
    mujoco.mj_forward(model, simulation.mj_data)
    targets = np.clip(
        simulation.mj_data.actuator_length,
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    simulation.data.ctrl[0].copy_(torch.as_tensor(targets, dtype=torch.float32))
    return simulation


def sync_viewer_data(simulation: Simulation) -> None:
    """Copy Warp state for rendering only; mjlab performs all integration."""
    data = simulation.mj_data
    data.qpos[:] = simulation.data.qpos[0].numpy()
    data.qvel[:] = simulation.data.qvel[0].numpy()
    data.ctrl[:] = simulation.data.ctrl[0].numpy()
    data.time = float(simulation.data.time[0])
    mujoco.mj_forward(simulation.mj_model, data)


def run(steps: int, *, viewer: bool = False, motion: str = "hold") -> dict:
    simulation = make_simulation()
    model = simulation.mj_model
    targets = simulation.data.ctrl[0].clone()
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_knee_actuator")
    if viewer:
        from mujoco.viewer import launch_passive

        context = launch_passive(model, simulation.mj_data)
    else:
        context = nullcontext(None)
    completed = 0
    start = time.monotonic()
    with context as window:
        for step in range(steps):
            if window is not None and not window.is_running():
                break
            tick = time.monotonic()
            if motion == "sine":
                target = float(targets[knee]) + 0.2 * np.sin(
                    2 * np.pi * step * model.opt.timestep
                )
                simulation.data.ctrl[0, knee] = float(
                    np.clip(target, *model.actuator_ctrlrange[knee])
                )
            simulation.step()
            completed += 1
            if not (
                torch.isfinite(simulation.data.qpos).all()
                and torch.isfinite(simulation.data.qvel).all()
            ):
                raise RuntimeError(f"Non-finite physics state at step {completed}")
            if window is not None:
                with window.lock():
                    sync_viewer_data(simulation)
                window.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - tick)))
    return {
        "backend": "mjlab",
        "device": "cpu",
        "motion": motion,
        "steps": completed,
        "simulation_seconds": float(simulation.data.time[0]),
        "wall_seconds": time.monotonic() - start,
        "base_position": simulation.data.qpos[0, :3].tolist(),
        "finite": True,
    }


def run_policy(checkpoint: Path, steps: int, *, viewer: bool = False) -> dict:
    """Play a trusted training checkpoint using the training environment."""
    from rsl_rl.runners import OnPolicyRunner

    from mjlab_env import CatbotMjlabEnv

    config = json.loads(checkpoint.with_name("config.json").read_text())
    torch.set_num_threads(1)
    environment = CatbotMjlabEnv(
        1,
        "cpu",
        seed=config["environment"]["seed"],
        max_episode_length=config["environment"]["max_episode_length"],
    )
    runner = OnPolicyRunner(environment, config["runner"], device="cpu")
    runner.load(str(checkpoint), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    simulation = environment.sim
    sync_viewer_data(simulation)
    if viewer:
        from mujoco.viewer import launch_passive

        context = launch_passive(simulation.mj_model, simulation.mj_data)
    else:
        context = nullcontext(None)
    completed = episodes = 0
    total_reward = 0.0
    with torch.inference_mode(), context as window:
        for _ in range(steps):
            if window is not None and not window.is_running():
                break
            tick = time.monotonic()
            action = policy(environment.get_observations())
            if not torch.isfinite(action).all():
                raise RuntimeError("Policy produced non-finite actions")
            _, reward, done, _ = environment.step(action)
            completed += 1
            episodes += int(done[0])
            total_reward += float(reward[0])
            if window is not None:
                with window.lock():
                    sync_viewer_data(simulation)
                window.sync()
                time.sleep(max(0, environment.dt - (time.monotonic() - tick)))
    return {
        "checkpoint": str(checkpoint),
        "device": "cpu",
        "control_steps": completed,
        "episodes_completed": episodes,
        "total_reward": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Trusted model_*.pt checkpoint with adjacent config.json.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Physics steps, or policy control steps (0.02 seconds) with a checkpoint.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Native viewer (automatically uses mjpython on macOS).",
    )
    parser.add_argument("--motion", choices=("hold", "sine"), default="hold")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.checkpoint:
        if (
            not args.checkpoint.is_file()
            or not args.checkpoint.with_name("config.json").is_file()
        ):
            parser.error("Checkpoint and adjacent config.json must exist")
        if args.motion != "hold":
            parser.error("--motion sine cannot be combined with a checkpoint")
    if args.viewer and sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        # uv Python uses @rpath/libpython; mjpython changes the executable path.
        # Supply the interpreter's library directory to dyld before relaunching.
        env = os.environ.copy()
        library_dir = sysconfig.get_config_var("LIBDIR")
        if library_dir:
            existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
            env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{library_dir}:{existing}"
        launcher = Path(sys.executable).parent / "mjpython"
        os.execve(
            sys.executable,
            [
                sys.executable,
                str(launcher),
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            env,
        )
    if args.checkpoint:
        result = run_policy(args.checkpoint, args.steps, viewer=args.viewer)
    else:
        result = run(args.steps, viewer=args.viewer, motion=args.motion)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
