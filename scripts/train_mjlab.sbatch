#!/bin/bash -l
#SBATCH --job-name=catbot-ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

# Submit from the repository root; account and partition are supplied to sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
# User-owned shell configuration, kept outside the shared checkout.
catbot_config="$HOME/.config/catbot/training.env"
if [[ -f "$catbot_config" ]]; then
    source "$catbot_config"
fi
test -x .venv/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Preserve Slurm's CUDA_VISIBLE_DEVICES assignment. Do not use physical GPU IDs.
nvidia-smi
ldd --version
.venv/bin/python - <<'PY'
import torch
import warp as wp
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "No usable CUDA device in this allocation"
print("GPU:", torch.cuda.get_device_name(0))
wp.init()
assert wp.get_device("cuda:0").is_cuda
PY

run_dir="runs/mjlab/slurm-${SLURM_JOB_ID}"
# The trainer refuses to overwrite an existing run directory.
args=(--device cuda:0 --num-envs "${NUM_ENVS:-256}"
      --iterations "${ITERATIONS:-1000}" --save-interval "${SAVE_INTERVAL:-25}"
      --log-dir "$run_dir" --logger "${TRAIN_LOGGER:-wandb}"
      --wandb-project "${WANDB_PROJECT:-catbot}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    args+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${RESUME:-}" ]]; then
    args+=(--resume "$RESUME")
fi
srun .venv/bin/python mjlab_train.py "${args[@]}"
