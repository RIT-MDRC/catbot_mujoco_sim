# Catbot PPO training on RIT Research Computing

The new trainer uses **mjlab 1.6.0 + MuJoCo Warp + RSL-RL 5.4.2 PPO**. Both
batched physics and the actor/critic run on one NVIDIA GPU. `uv run train` and
`uv run mjlab-train` select this trainer; Stable-Baselines3 is retained only for
the legacy `train-sb3`/`rollout` commands and old `.zip` checkpoints.

This guide was checked against RIT documentation on September 28, 2026. Local
CPU tests exercise physics, PPO updates, checkpoint saving/loading and resets.
No job has been submitted to RIT from this workspace; account access, Linux
dependencies, GPU driver compatibility, speed and learned gait quality still
need verification on your allocation. Start with the short debug job below.

## 1. Connect and choose an allocation

From your Mac, connect to the cluster where your project has an allocation:

```sh
ssh YOUR_RIT_USERNAME@sporcsubmit.rc.rit.edu
# Or, for a TIGRIS allocation:
ssh YOUR_RIT_USERNAME@tigris.rc.rit.edu
```

RIT documents Duo push authentication for password-based SSH. Once connected:

```sh
my-accounts
sinfo -o '%P %l %G'
```

Use an account returned by `my-accounts`, not your username unless they really
are identical. Confirm GPU availability and time limits with `sinfo`. RIT
documents `sporc` for SPORC research jobs and `tigris` for TIGRIS research jobs;
`debug` is for troubleshooting, not production training. The examples below
use SPORC. Select the partition your account can access on your chosen cluster.
Do not train on the login/submit node.

Sources: [RIT getting started](https://research-computing.git-pages.rit.edu/docs/index.html),
[Slurm tutorial](https://research-computing.git-pages.rit.edu/docs/slurm_quick_start_tutorial.html).

## 2. Transfer the project

Use your project's shared directory when available, typically
`/shared/rc/PROJECT_NAME`. A private experiment can live under your home directory.
Set the actual path on the cluster and create it:

```sh
export CATBOT_DIR=/shared/rc/YOUR_PROJECT/catbot_mujoco
mkdir -p "$CATBOT_DIR"
```

In a separate Mac terminal, copy this checkout, including the newly added files.
Replace both the username and remote directory. This excludes the Mac virtual
environment and local checkpoints and does not delete remote files:

```sh
rsync -av --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
  --exclude='.ruff_cache' --exclude='.DS_Store' --exclude='runs*' --exclude='wandb' \
  ./ YOUR_RIT_USERNAME@sporcsubmit.rc.rit.edu:/shared/rc/YOUR_PROJECT/catbot_mujoco/
```

Use `tigris.rc.rit.edu` instead when working on TIGRIS. A Git clone of a commit
containing all these changes is also fine. Recreate `.venv` on Linux; never copy
the Mac environment. Keep checkpoints in home/project storage, not `/tmp` or
`/scratch`, which are temporary. RIT storage is not a backup: copy valuable
checkpoints elsewhere. See [RIT storage](https://research-computing.git-pages.rit.edu/docs/storage_tutorial.html).

## 3. Create the Linux environment once

RIT supports Spack environments and permits user-managed software installs.
This recipe uses a project-local uv environment to honor `uv.lock`; it does not
assume the cluster's `default-ml` stack matches our versions. User-managed
installs have a different support level from RC-maintained Spack environments.
If you need an RC-supported setup, request one with `pyproject.toml` and
`uv.lock`; activate the **actual environment name RC provides** with
`spack env activate NAME` in your setup and batch script. Do not copy old Spack
package hashes from tutorials or mix a Spack PyTorch with this virtualenv.
See [RIT software tutorial](https://research-computing.git-pages.rit.edu/docs/software_tutorial.html).

In a clean cluster shell:

```sh
cd "$CATBOT_DIR"
command -v uv
# Only if uv is absent (user installation, no sudo):
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/catbot-install-uv.sh
sh /tmp/catbot-install-uv.sh --no-modify-path
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked
mkdir -p slurm-logs
```

Run the installer lines only when `command -v uv` found nothing. No shell
startup-file modification is required. The job script uses `.venv/bin/python`
directly, so jobs neither resolve dependencies nor download packages.

**Compatibility gate:** this project currently requires Python 3.14. Its lockfile
selects PyTorch 2.14.0 with CUDA 13 libraries on Linux, and the Linux PyTorch
wheels require glibc 2.28 or newer. This is stricter than mjlab's general CUDA
12.4+ recommendation. Check `ldd --version` on the compute node and the
`nvidia-smi`/PyTorch preflight in the debug job. A CUDA module alone cannot
upgrade the host driver. If the node is too old, use an eligible newer node or
ask RC for a compatible environment/container and agree on dependency versions
before regenerating the lockfile. Do not silently substitute a CPU PyTorch
wheel. We have not verified which RIT nodes meet these requirements.

`uv sync` needs network access and sufficient storage for the Linux CUDA wheels.
If installation fails on cluster policy, networking or OS compatibility, retain
the error and ask RC; do not install packages repeatedly inside training jobs.
The first Warp run also builds a kernel cache and can take longer than later runs.
See [mjlab installation](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html).

## 4. Submit a short GPU smoke test

W&B is the default logger. Complete the login setup in section 6 before
submitting, or set `export WANDB_MODE=offline` to sync later. For local-only
TensorBoard logging, set `export TRAIN_LOGGER=tensorboard` instead.

From the repository root on the cluster:

```sh
cd "$CATBOT_DIR"
mkdir -p slurm-logs
export CATBOT_ACCOUNT=YOUR_SLURM_ACCOUNT
sbatch --account="$CATBOT_ACCOUNT" --partition=debug --time=00:15:00 \
  --export=ALL,NUM_ENVS=4,ITERATIONS=2,SAVE_INTERVAL=1 scripts/train_mjlab.sbatch
```

If your cluster/account has no GPU-enabled `debug` partition, use an authorized
GPU partition with the same short limit. The script requests one generic GPU,
four CPUs and 16 GB RAM. These are starting estimates, not measured requirements.
If you need a specific available GPU, override with e.g. `--gres=gpu:a100:1`
after checking `sinfo`. Keep `--ntasks=1`; this is one batched training process,
not one Slurm task per simulation world.

Note the job ID printed by `sbatch`:

```sh
squeue --me
tail -f slurm-logs/catbot-ppo-JOB_ID.out
cat slurm-logs/catbot-ppo-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Success means CUDA preflight passes, the log reports PPO learning iterations
with finite losses, Slurm reports `COMPLETED`/exit code zero, and the directory
`runs/mjlab/slurm-JOB_ID/` contains `config.json`, `model.xml`, TensorBoard events,
and `model_0.pt` / `model_1.pt`. The model files contain actor, critic,
normalizers and optimizer state. A queued job or `nvidia-smi` alone does not
prove training works. Inspect stderr even after a successful run.

The script preserves Slurm's `CUDA_VISIBLE_DEVICES`; `cuda:0` means the first
GPU assigned to the job, not physical GPU zero. Training is headless and needs
no display, native viewer, or EGL rendering setup.

## 5. Start training

After the debug job succeeds:

```sh
sbatch --account="$CATBOT_ACCOUNT" --partition=sporc \
  --export=ALL,NUM_ENVS=256,ITERATIONS=1000,SAVE_INTERVAL=25 scripts/train_mjlab.sbatch
# Use --partition=tigris instead for the corresponding TIGRIS allocation.
```

This starts a fresh run. Each PPO iteration collects 24 control steps per world;
256 worlds × 24 × 1,000 iterations = 6,144,000 transitions. Each control step
advances 10 physics steps at 0.002 seconds. The script's four-hour limit is a
starting budget, not a completion-time estimate. Inspect throughput and GPU
memory before increasing worlds to 512 or 1,024. Lower `NUM_ENVS` on out-of-memory
errors. More worlds require no additional Slurm tasks or GPUs.

Checkpoints are written at iteration zero, every `SAVE_INTERVAL` iterations and
at normal completion. All checkpoints are retained. Slurm cancellation or timeout
can lose work since the last completed checkpoint; there is no automatic requeue
or termination-time save. Use `scancel JOB_ID` to stop a job. Do not reuse a run
directory: each submission writes `runs/mjlab/slurm-JOB_ID` and the trainer refuses
to overwrite an existing directory.

Resume from an actual complete checkpoint, in a new job/run directory:

```sh
export RESUME="$CATBOT_DIR/runs/mjlab/slurm-OLD_JOB_ID/model_975.pt"
sbatch --account="$CATBOT_ACCOUNT" --partition=sporc \
  --export=ALL,NUM_ENVS=256,ITERATIONS=1000,SAVE_INTERVAL=25 scripts/train_mjlab.sbatch
unset RESUME
```

`ITERATIONS` means **additional** updates on resume. Actor, critic, normalizers,
optimizer and iteration count are restored; simulation and RNG states restart,
so this is not bit-for-bit continuation. Use the same model/configuration and
trusted checkpoints from this trainer. SB3 `.zip` files cannot be resumed here.

## 6. W&B dashboards and cloud checkpoints

The trainer and Slurm script use W&B by default. Select `--logger tensorboard`
for local-only logging, or set `TRAIN_LOGGER=tensorboard` for Slurm jobs.
W&B mode writes the same local checkpoints and TensorBoard events, logs metrics
to a browser dashboard, and uploads each completed checkpoint plus `config.json`
and `model.xml` to the run's **Files** tab. These are run files, not versioned
W&B Artifacts. All checkpoints are retained locally and remotely; monitor your
W&B storage quota and choose `SAVE_INTERVAL` accordingly. Uploads are asynchronous:
a saved local checkpoint does not prove its cloud upload completed.

Local validation covered a two-iteration CPU run with the actual W&B SDK in
offline mode, both checkpoint files and metadata staged for sync, and local
TensorBoard events. Authenticated cloud uploads and RIT GPU jobs remain unverified.

After transferring the updated project, on the cluster's login/submit node:

```sh
cd "$CATBOT_DIR"
uv sync --locked
uv run wandb login
export WANDB_ENTITY=YOUR_WANDB_USER_OR_TEAM
export WANDB_PROJECT=catbot
export WANDB_MODE=online
export TRAIN_LOGGER=wandb
```

Create/select your W&B account or team and a private project in the W&B website.
Paste the API key at the interactive login prompt; do not put it in this repo,
the batch script, or an `sbatch --export` argument. Login credentials normally
live in your home directory and must be accessible from the compute node.
`WANDB_ENTITY` is the W&B user/team slug, not your RIT username. No TensorBoard
server or inbound port is needed for the W&B dashboard.

First submit a short job with W&B enabled:

```sh
sbatch --account="$CATBOT_ACCOUNT" --partition=debug --time=00:15:00 \
  --export=ALL,NUM_ENVS=4,ITERATIONS=2,SAVE_INTERVAL=1 scripts/train_mjlab.sbatch
```

`--export=ALL` carries these settings into the job. Use an authorized GPU partition
if `debug` is unavailable. Verify the printed W&B run URL, metric charts, and
`config.json`, `model.xml`, `model_0.pt`, and `model_1.pt` in Files before relying
on cloud storage. Then submit the normal training command from section 5.
Each submission, including `RESUME`, creates a new W&B run named `slurm-JOB_ID`;
model state resumes, but the old W&B run is not reopened. Leave `WANDB_RUN_ID`
and `WANDB_RESUME` unset. To use TensorBoard-only, `export TRAIN_LOGGER=tensorboard`.

For a direct invocation (including a local CPU smoke test):

```sh
WANDB_MODE=offline uv run mjlab-train --device cpu --num-envs 2 \
  --steps-per-env 4 --iterations 2 --save-interval 1 \
  --logger wandb --wandb-project catbot
```

### Compute nodes without outbound internet

RIT compute-node connectivity to W&B has not been verified. If online
initialization fails, set `export WANDB_MODE=offline` before submitting a new
job. Offline logging requires no login on the compute node. It preserves metrics
and file-upload records under `runs/mjlab/slurm-JOB_ID/wandb/offline-run-*`, but
does not update the cloud dashboard while training.

After the job ends, sync from a network-connected login node:

```sh
uv run wandb login
uv run wandb sync runs/mjlab/slurm-JOB_ID/wandb/offline-run-*
```

Keep the whole run directory, including the original checkpoint files, until
sync finishes and you verify the files online: W&B's staging directory uses
symlinks to saved files. If syncing from your Mac instead, transfer the whole
run with `rsync -avL` to dereference those links. Interrupted jobs may leave
pending uploads; keep local data and inspect W&B/Slurm logs. There is no
automatic online-to-offline fallback or guaranteed upload on Slurm termination.

### Download a checkpoint on your Mac

Download `model_N.pt`, `config.json`, and `model.xml` from the run's Files tab
into the same new directory under `runs/mjlab/`. Alternatively, authenticate
with `uv run wandb login` and use the API (replace the run ID and filename):

```sh
uv run python - <<'PY'
from pathlib import Path
import wandb

destination = Path("runs/mjlab/downloaded-run")
destination.mkdir(parents=True, exist_ok=False)
run = wandb.Api().run("YOUR_ENTITY/catbot/YOUR_WANDB_RUN_ID")
for name in ("model_975.pt", "config.json", "model.xml"):
    run.file(name).download(root=str(destination), replace=False)
PY
uv run mjlab-rollout runs/mjlab/downloaded-run/model_975.pt --viewer --steps 1500
```

To resume on RIT, download the files there and set `RESUME` to the downloaded
checkpoint as in section 5. Only load trusted checkpoints.

References: [W&B file uploads](https://docs.wandb.ai/ref/python/experiments/run/),
[W&B CLI](https://docs.wandb.ai/ref/cli/),
[W&B file downloads](https://docs.wandb.ai/models/ref/python/public-api/file).

## 7. Inspect results on your Mac with TensorBoard

Copy a completed run back (substitute the run ID and remote path):

```sh
mkdir -p runs/mjlab
rsync -av YOUR_RIT_USERNAME@sporcsubmit.rc.rit.edu:/shared/rc/YOUR_PROJECT/catbot_mujoco/runs/mjlab/slurm-JOB_ID/ \
  runs/mjlab/slurm-JOB_ID/
uv run tensorboard --logdir runs/mjlab --host 127.0.0.1
```

Open the localhost URL printed by TensorBoard. Compare episode returns/lengths,
tracking, uprightness, foot drag, failures and PPO losses. Better returns alone
do not establish a useful gait. The old `uv run rollout` reads SB3 checkpoints,
not these RSL-RL `.pt` files. Play a trusted RSL-RL checkpoint on the CPU with:

```sh
uv run mjlab-rollout runs/mjlab/YOUR_RUN/model_1.pt --viewer --steps 1500
```

Keep `config.json` beside the checkpoint. Playback uses the training environment,
including action smoothing and automatic resets after falls. Each policy step
is 0.02 simulated seconds; omit `--viewer` for a headless check. Without a
checkpoint, `mjlab-rollout` still tests the model without a learned policy.

For a small local training smoke test without a GPU:

```sh
uv run mjlab-train --logger tensorboard --device cpu --num-envs 2 --steps-per-env 4 \
  --iterations 2 --save-interval 1
uv run python -m unittest discover -s tests
```

CPU mode exists for validation, not the intended large training workload.

## Training design and current limits

- Actor and critic: separate 256/128/64 ELU MLPs with observation normalization;
  Gaussian actor with initial standard deviation 0.5. PPO uses mjlab's defaults:
  five epochs, four minibatches, adaptive learning rate starting at 0.001,
  clipping 0.2, discount 0.99, GAE lambda 0.95. These are starting settings,
  not tuned hyperparameters.
- The robot/scene MJCF is unchanged. All 16 action slots remain, including locked
  hip-z; actions are clipped, smoothed with the original 0.8 coefficient and
  mapped to actuator ranges. Observations retain the original 66-value layout.
- Reward weights and 20-degree tilt/0.14 m fall thresholds follow `CatbotEnv`.
  Foot contacts and rewards are batched on-device. Episodes auto-reset per world;
  time limits are distinguished from falls for PPO bootstrapping.
- This initial training environment uses fixed masses/friction and reference
  joint poses on reset. It randomizes base position, velocity and command.
  Legacy mass/friction/joint-pose randomization has not been migrated. This is
  a simulation training baseline, not a hardware-ready policy.
- Only single-GPU execution is wired up. No multi-node/DDP training, curriculum,
  or automatic hyperparameter search is implemented. W&B is optional;
  TensorBoard-only and offline runs do not require an external-service login.

Reference: [RIT Slurm quick reference](https://research-computing.git-pages.rit.edu/docs/slurm_reference.html).
