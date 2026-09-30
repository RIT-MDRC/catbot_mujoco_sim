"""Download a selected W&B checkpoint and play it in the local MuJoCo viewer."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="W&B run path: ENTITY/PROJECT/RUN_ID")
    parser.add_argument("checkpoint", help="Checkpoint filename, e.g. model_999.pt")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    parts = args.run.split("/")
    if len(parts) != 3 or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts
    ):
        parser.error("run must be ENTITY/PROJECT/RUN_ID")
    if not re.fullmatch(r"model_\d+\.pt", args.checkpoint):
        parser.error("checkpoint must be a filename such as model_999.pt")
    if args.steps < 1:
        parser.error("--steps must be positive")

    import wandb

    root = Path(__file__).resolve().parent
    destination = root / "runs" / "mjlab" / "wandb" / Path(*parts)
    run = wandb.Api().run(args.run)
    names = (args.checkpoint, "config.json", "model.xml")
    files = {file.name: file for file in run.files() if file.name in names}
    missing = set(names) - files.keys()
    if missing:
        parser.error(f"Run is missing files: {', '.join(sorted(missing))}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"Downloading {name}", flush=True)
        # Refresh on each invocation, including checkpoints from active runs.
        files[name].download(root=str(destination), replace=True).close()
    checkpoint = destination / args.checkpoint
    print(f"Playing {checkpoint}", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "mjlab_rollout.py"),
            str(checkpoint),
            "--viewer",
            "--steps",
            str(args.steps),
        ],
        cwd=root,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
