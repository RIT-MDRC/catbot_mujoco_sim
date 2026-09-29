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
  --exclude='.ruff_cache' --exclude='.DS_Store' --exclude='runs*' \
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

## 6. Inspect results on your Mac

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
not these RSL-RL `.pt` files; the checkpoint-free `mjlab-rollout` still tests the
model without a learned policy.

For a small local training smoke test without a GPU:

```sh
uv run mjlab-train --device cpu --num-envs 2 --steps-per-env 4 \
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
  automatic hyperparameter search, or external experiment-service login is needed.

Reference: [RIT Slurm quick reference](https://research-computing.git-pages.rit.edu/docs/slurm_reference.html).
