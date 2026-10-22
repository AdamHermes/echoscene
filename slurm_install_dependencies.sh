#!/bin/bash
#SBATCH --job-name=echoscene_install
#SBATCH --output=logs/install_dependencies_%j.out
#SBATCH --error=logs/install_dependencies_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Submit from the echoscene repository root:
#   mkdir -p logs
#   sbatch slurm_install_dependencies.sh
# The echoscene Conda environment must already exist with Python 3.8.
# Compute nodes need Internet access for package downloads.
set -eo pipefail
trap 'echo "Installation failed at line $LINENO. Check the job logs." >&2' ERR

cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the echoscene repository root}}"
if [[ ! -f requirements.txt ]]; then
    echo "requirements.txt not found. Submit from the echoscene repository root." >&2
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found. Load Conda before submitting this job." >&2
    exit 1
fi
conda activate echoscene
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python - <<'PY'
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
if sys.version_info[:2] != (3, 8):
    raise SystemExit('This PyTorch3D wheel requires Python 3.8. Create echoscene with: conda create -n echoscene python=3.8 -y')
PY

# Install the requested PyTorch version first so CLIP uses this installation.
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
python -m pip install einops omegaconf tensorboardx open3d
python -m pip install ftfy regex tqdm
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

python -m pip check
python - <<'PY'
import open3d
import torch
import torchvision
import torchaudio
import einops
import omegaconf
import tensorboardX
import ftfy
import regex
import tqdm
import clip
import pytorch3d
from pytorch3d import _C

print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('PyTorch3D:', pytorch3d.__version__)
assert torch.__version__ == '1.11.0+cu113', 'Unexpected PyTorch version'
assert torch.cuda.is_available(), 'CUDA is unavailable on the allocated GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA tensor check:', (torch.ones(1, device='cuda') + 1).item())
print('Dependency installation and import checks completed successfully.')
PY
