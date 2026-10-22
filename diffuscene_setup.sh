#!/bin/bash
# =====================================================================
# DiffuScene - Setup & Run trên SoC Compute Cluster
# =====================================================================
# Nguyên tắc chia node:
#   - LOGIN NODE (xlogin): git clone, tạo conda env, cài đặt python
#     packages thuần (không cần compile CUDA), tải model/dataset.
#     Đây là các việc "nhẹ", không cần GPU.
#   - COMPUTE NODE (srun): build các extension CUDA (nvcc), và chạy
#     script generate/eval thật sự cần GPU.
#     Build extension CUDA nên làm ở đây vì login node không có GPU
#     driver, dễ build sai kiến trúc GPU hoặc thiếu nvcc đúng bản.
# =====================================================================


# #####################################################################
# PHẦN 1 — CHẠY TRÊN LOGIN NODE (xlogin)
# #####################################################################

# --- 1.1 Clone source code -------------------------------------------
cd ~
git clone https://github.com/AdamHermes/DiffuScene.git
cd DiffuScene

# --- 1.2 Cài Miniconda (bỏ qua nếu đã có conda) ----------------------
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
source ~/soc_cluster_cheatsheet.sh
# --- 1.3 Tạo conda environment ---------------------------------------
conda create -n diffuscene -c conda-forge --override-channels python=3.8
conda activate diffuscene

# --- 1.4 Cài các package Python thuần (không cần compile CUDA) ------
pip install \
    cython==0.29.32 \
    numpy==1.21.3 \
    networkx==2.5.1 \
    pyrr==0.10.3 \
    pyyaml==5.3.1 \
    pillow==7.2.0 \
    scipy==1.8.1 \
    tqdm==4.64.1 \
    trimesh==3.12.7 \
    matplotlib==3.3.1 \
    simple_3dviz==0.4.0 \
    num2words \
    nltk \
    einops_exts \
    kornia==0.6.9 \
    rotary_embedding_torch \
    transformers==4.25.1 \
    pyvista==0.43.3 \
    torchtext \
    seaborn \
    wandb \
    shapely \
    pytorch-fast-transformers

pip install git+https://github.com/openai/CLIP.git

python -c "import nltk; nltk.download('cmudict')"

# --- 1.5 Tải pretrained models + dataset đã preprocess ---------------
mkdir -p models datasets
pip install --upgrade gdown

# Thay ID bằng đúng file ID Google Drive của bạn (nếu dùng cách này):
# gdown <FILE_ID_pretrained_models> -O models/pretrained_diffusion.zip
# gdown <FILE_ID_3d_front_processed> -O datasets/3d_front_processed.zip

unzip -q models/pretrained_diffusion.zip -d models/
unzip -q datasets/3d_front_processed.zip -d datasets/


# #####################################################################
# PHẦN 2 — CHẠY TRÊN COMPUTE NODE (sau khi srun vào GPU node)
# #####################################################################
# Trước tiên, xin GPU node từ login node:
#   srun --gpus=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
#
# Sau khi vào compute node, chạy tiếp các lệnh dưới đây:

cd ~/DiffuScene
conda activate diffuscene

# --- 2.1 Cài torch đúng bản CUDA -------------------------------------
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# --- 2.2 Build các extension CUDA của repo ---------------------------
python setup.py build_ext --inplace
pip install -e .

# --- 2.3 Cài + build ChamferDistancePytorch ---------------------------
cd ChamferDistancePytorch/chamfer3D
python setup.py install
cd ~/DiffuScene

# --- 2.4 Cài thêm tool phục vụ render (xvfb ảo, không cần màn hình) --
# Lưu ý: apt-get cần quyền sudo - nếu không có sudo trên compute node,
# bỏ qua bước này và render mà không dùng virtual display, hoặc báo
# admin cluster cài sẵn xvfb hệ thống.
# sudo apt-get install -y xvfb
pip install pyvirtualdisplay

# --- 2.5 Chạy thử generate --------------------------------------------
cd scripts

python - <<'EOF'
from pyvirtualdisplay import Display
display = Display(visible=0, size=(512, 512))
display.start()
EOF

exp_dir="../models"
exp_name="pretrained_diffusion/bedrooms_uncond"
config="../config/uncond/diffusion_bedrooms_instancond_lat32_v.yaml"
threed_future="../datasets/3d_front_processed/threed_future_model_bedroom.pkl"
weight_file="${exp_dir}/${exp_name}/model_30000"
output_dir="${exp_dir}/${exp_name}/gen_top2down_notexture_nofloor"

python -u generate_diffusion.py "${config}" "${output_dir}" "${threed_future}" \
    --weight_file "${weight_file}" \
    --n_sequences 1 \
    --fix_order \
    --start_index 0 \
    --render_top2down \
    --save_mesh \
    --no_texture \
    --without_floor \
    --clip_denoised \
    --retrive_objfeats

# --- 2.6 Copy kết quả ra ngoài thư mục lưu lâu dài (~/) --------------
cp -r "../models/pretrained_diffusion/bedrooms_uncond/gen_top2down_notexture_nofloor" \
      "~/DiffuScene_results/run20_3"