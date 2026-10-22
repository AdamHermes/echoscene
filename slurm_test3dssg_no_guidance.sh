#!/bin/bash
#SBATCH --job-name=3dssg_no_guidance
#SBATCH --output=logs/test3dssg_no_guidance_%A_%a.out
#SBATCH --error=logs/test3dssg_no_guidance_%A_%a.err
#SBATCH --array=1
#SBATCH --nodelist=xgpg4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Submit from the repository root after: mkdir -p logs
# Run task 1 of the fixed six partitions used by slurm_test3dssg.sh.
# Override the time limit with sbatch --time=... if the partition permits it.
set -eo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
REPO_DIR="$PWD"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Load Conda before submitting this job." >&2
    exit 1
fi
conda activate echoscene
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
JSON_DIR="$REPO_DIR/test3DSSG/adapted_scenes"
TOTAL_SCENES="$(python -c 'import glob, os, sys; print(len(glob.glob(os.path.join(sys.argv[1], "*.json"))))' "$JSON_DIR")"
TASK_ID="$SLURM_ARRAY_TASK_ID"
if (( TASK_ID < 0 || TASK_ID >= 6 )); then
    echo "Task ID must be between 0 and 5." >&2
    exit 1
fi
START_IDX=$((TOTAL_SCENES * TASK_ID / 6))
END_IDX=$((TOTAL_SCENES * (TASK_ID + 1) / 6))
MAX_SAMPLES=$((END_IDX - START_IDX))
if (( MAX_SAMPLES == 0 )); then
    echo "No scenes assigned to task $TASK_ID."
    exit 0
fi
OUTPUT_DIR="$REPO_DIR/released_full_model/vis_3dssg/2050/no_guidance/array_${SLURM_ARRAY_JOB_ID}/task_${TASK_ID}"
echo "Task $TASK_ID: indices [$START_IDX, $END_IDX) of $TOTAL_SCENES scenes ($MAX_SAMPLES scenes)"
echo "Node: ${SLURMD_NODENAME:-unknown}; output: $OUTPUT_DIR"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print("GPU:", torch.cuda.get_device_name(0))'
cd "$REPO_DIR/scripts"
srun python -u eval_3dssg.py \
    --json_dir "$JSON_DIR" \
    --start_idx "$START_IDX" --max_samples "$MAX_SAMPLES" \
    --exp "$REPO_DIR/released_full_model" \
    --dataset "$REPO_DIR/FRONT" \
    --epoch 2050 --room_type all \
    --render_type echoscene --gen_shape True --save_3d True \
    --disable_physical_guidance \
    --output_dir "$OUTPUT_DIR"
