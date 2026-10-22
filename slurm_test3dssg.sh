#!/bin/bash
#SBATCH --job-name=echoscene_test3dssg
#SBATCH --output=logs/test3dssg_%A_%a.out
#SBATCH --error=logs/test3dssg_%A_%a.err
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Submit from the repository root:
#   mkdir -p logs
#   sbatch slurm_test3dssg.sh
# Activate the environment before submission, or let this script activate it.
# Optional overrides: CONDA_ENV, REPO_DIR, JSON_DIR, EXP_DIR, DATASET_DIR, EPOCH.
set -eo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
cd "$REPO_DIR"
REPO_DIR="$PWD"

CONDA_ENV="${CONDA_ENV:-echoscene}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Activate the Conda environment before running sbatch." >&2
        exit 1
    fi
    conda activate "$CONDA_ENV"
fi
set -u

JSON_DIR="${JSON_DIR:-$REPO_DIR/test3DSSG/adapted_scenes}"
EXP_DIR="${EXP_DIR:-$REPO_DIR/released_full_model}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/FRONT}"
EPOCH="${EPOCH:-2050}"
# Resolve relative overrides before switching to scripts/ for model config paths.
JSON_DIR="$(cd "$JSON_DIR" && pwd)"
EXP_DIR="$(cd "$EXP_DIR" && pwd)"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"

TOTAL_SCENES="$(python -c 'import glob, os, sys; print(len(glob.glob(os.path.join(sys.argv[1], "*.json"))))' "$JSON_DIR")"
TASK_OFFSET=$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))
TASK_COUNT="$SLURM_ARRAY_TASK_COUNT"
START_IDX=$((TOTAL_SCENES * TASK_OFFSET / TASK_COUNT))
END_IDX=$((TOTAL_SCENES * (TASK_OFFSET + 1) / TASK_COUNT))
MAX_SAMPLES=$((END_IDX - START_IDX))
if (( MAX_SAMPLES == 0 )); then
    echo "Task $SLURM_ARRAY_TASK_ID has no scenes to process (total: $TOTAL_SCENES)."
    exit 0
fi

OUTPUT_DIR="$EXP_DIR/vis_3dssg/$EPOCH/array_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Task $SLURM_ARRAY_TASK_ID: indices [$START_IDX, $END_IDX) of $TOTAL_SCENES scenes"
echo "Output: $OUTPUT_DIR"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print("GPU:", torch.cuda.get_device_name(0))'

cd "$REPO_DIR/scripts"
srun python -u eval_3dssg.py \
    --json_dir "$JSON_DIR" \
    --start_idx "$START_IDX" \
    --max_samples "$MAX_SAMPLES" \
    --exp "$EXP_DIR" \
    --dataset "$DATASET_DIR" \
    --epoch "$EPOCH" \
    --room_type all \
    --render_type echoscene \
    --gen_shape True \
    --save_3d True \
    --output_dir "$OUTPUT_DIR"
