"""Train Catbot with batched mjlab physics and RSL-RL PPO."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner

from catbot_env import CatbotEnv
from mjlab_env import CatbotMjlabEnv


def runner_config(
    steps=24,
    save_interval=50,
    *,
    logger="wandb",
    wandb_project="catbot",
    wandb_entity=None,
):
    config = asdict(RslRlOnPolicyRunnerCfg())
    config.update(
        num_steps_per_env=steps,
        save_interval=save_interval,
        logger=(
            {
                "class_name": "training_wandb:CatbotWandbWriter",
                "project": wandb_project,
                "entity": wandb_entity,
            }
            if logger == "wandb"
            else "tensorboard"
        ),
    )
    config["actor"]["hidden_dims"] = [256, 128, 64]
    config["critic"]["hidden_dims"] = [256, 128, 64]
    config["actor"]["obs_normalization"] = True
    config["critic"]["obs_normalization"] = True
    config["actor"]["distribution_cfg"]["init_std"] = 0.5
    for name in ("actor", "critic"):
        for key in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            config[name].pop(key)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Additional PPO updates, including on resume.",
    )
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logger", choices=("tensorboard", "wandb"), default="wandb"
    )
    parser.add_argument("--wandb-project", default="catbot")
    parser.add_argument(
        "--wandb-entity", help="W&B user or team; defaults to W&B account settings."
    )
    parser.add_argument(
        "--log-dir", type=Path, help="New run directory; must not already exist."
    )
    parser.add_argument(
        "--resume", type=Path, help="Trusted RSL-RL model_*.pt checkpoint."
    )
    args = parser.parse_args()
    for key in ("num_envs", "iterations", "steps_per_env", "save_interval"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.num_envs * args.steps_per_env < 4:
        parser.error("PPO needs at least 4 samples per update")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable. Request a GPU allocation and check the PyTorch/driver installation; use --device cpu only for local smoke tests."
        )
    if args.resume and not args.resume.is_file():
        parser.error(f"Missing checkpoint: {args.resume}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(1)
    log_dir = args.log_dir or Path("runs/mjlab") / datetime.now(UTC).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    environment = CatbotMjlabEnv(args.num_envs, args.device, args.seed)
    config = runner_config(
        args.steps_per_env,
        args.save_interval,
        logger=args.logger,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    config["seed"] = args.seed
    config["max_iterations"] = args.iterations
    (log_dir / "config.json").write_text(
        json.dumps(
            {"environment": environment.cfg, "runner": config, "arguments": vars(args)},
            indent=2,
            default=str,
        )
    )
    (log_dir / "model.xml").write_text(CatbotEnv._render_mjcf())
    runner = OnPolicyRunner(
        environment, deepcopy(config), str(log_dir), device=args.device
    )
    if args.resume:
        runner.load(str(args.resume), map_location=args.device)
        # RSL-RL stores the last completed zero-based iteration.
        runner.current_learning_iteration += 1
    print(f"Run directory: {log_dir.resolve()}", flush=True)
    try:
        runner.learn(num_learning_iterations=args.iterations)
    except BaseException:
        if args.logger == "wandb" and runner.logger.writer is not None:
            runner.logger.writer.stop(exit_code=1)
        raise


if __name__ == "__main__":
    main()
