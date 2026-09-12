"""Train a PPO policy for Catbot. Run with: uv run --extra train python train.py."""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

from catbot_env import CatbotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("runs/catbot_ppo"))
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help="Save checkpoints every N total environment timesteps; use 0 to disable.",
    )
    parser.add_argument(
        "--action-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable action smoothing during training (default: enabled).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to continue from (for example, runs/catbot_ppo.zip).",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--envs", type=int, default=4, help="Number of parallel environments to run."
    )
    args = parser.parse_args()
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as error:
        raise SystemExit(
            "Install training dependencies first: uv sync --extra train"
        ) from error
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable in this Python/PyTorch/macOS environment.")
    if args.resume is not None and not (
        args.resume.exists() or args.resume.with_suffix(".zip").exists()
    ):
        raise SystemExit(f"Resume checkpoint does not exist: {args.resume}")
    if args.checkpoint_freq < 0:
        raise SystemExit("--checkpoint-freq must be zero or a positive integer")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    checkpoint_prefix = f"{args.output.stem}_{run_id}"
    environment_count = args.envs
    environment = make_vec_env(
        CatbotEnv,
        n_envs=environment_count,
        seed=0,
        env_kwargs={"action_smoothing_enabled": args.action_smoothing},
    )
    if args.resume is None:
        model = PPO(
            "MlpPolicy",
            environment,
            verbose=1,
            device=args.device,
            tensorboard_log=str(args.output.parent),
        )
    else:
        model = PPO.load(args.resume, env=environment, device=args.device)
    callback = None
    if args.checkpoint_freq:
        from stable_baselines3.common.callbacks import BaseCallback

        # The callback runs once per vectorized environment step, which advances
        # the global counter by ``environment_count`` timesteps.

        class CheckpointCallback(BaseCallback):
            """Save regular checkpoints and retain only 100k-step milestones."""

            def __init__(
                self, save_freq: int, save_path: Path, name_prefix: str
            ) -> None:
                super().__init__(verbose=0)
                self.save_freq = save_freq
                self.save_path = save_path
                self.name_prefix = name_prefix

            def _on_step(self) -> bool:
                if self.num_timesteps % self.save_freq:
                    return True

                self.save_path.mkdir(parents=True, exist_ok=True)
                checkpoint = self.save_path / (
                    f"{self.name_prefix}_{self.num_timesteps}_steps"
                )
                self.model.save(checkpoint)

                if self.num_timesteps % 100_000 == 0:
                    checkpoint_pattern = re.compile(
                        rf"^{re.escape(self.name_prefix)}_(\d+)_steps\.zip$"
                    )
                    for path in self.save_path.iterdir():
                        match = checkpoint_pattern.fullmatch(path.name)
                        if match and int(match.group(1)) % 100_000:
                            path.unlink()
                return True

        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // environment_count, 1),
            save_path=args.output.parent / "checkpoints",
            name_prefix=checkpoint_prefix,
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=args.resume is None,
    )
    model.save(args.output)
    environment.close()


if __name__ == "__main__":
    main()
