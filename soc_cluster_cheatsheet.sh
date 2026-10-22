#!/bin/bash
# =====================================================================
# SoC Compute Cluster - Cheatsheet / Helper Script
# =====================================================================
# Cách dùng:
#   1. Các lệnh có nhãn [WINDOWS] chạy trên PowerShell của máy Windows.
#   2. Các lệnh có nhãn [CLUSTER] chạy sau khi đã SSH vào xlogin.
#   3. Có thể source file này trên cluster để dùng các function nhanh:
#        source ~/soc_cluster_cheatsheet.sh
#      rồi gõ: gpu_a100, gpu_any, myjobs, ...
# =====================================================================


# ---------------------------------------------------------------------
# [WINDOWS] Bước 1: SSH vào login node qua jump host
# ---------------------------------------------------------------------
# ssh -J bach66@stujump.comp.nus.edu.sg bach66@xlogin.comp.nus.edu.sg
# (sẽ hỏi password 2 lần: stujump rồi xlogin)


# ---------------------------------------------------------------------
# [CLUSTER] Bước 2: Kiểm tra tình trạng cluster trước khi xin GPU
# ---------------------------------------------------------------------
# sinfo -o "%N %G %t"          # xem node + GPU + trạng thái
# squeue -u $USER               # xem job của mình (PD = pending, R = running)


# ---------------------------------------------------------------------
# [CLUSTER] Bước 3: Xin một compute node có GPU (interactive)
# ---------------------------------------------------------------------
gpu_any() {
    # Xin bất kỳ GPU nào đang rảnh - dễ được cấp nhanh nhất
    srun --gpus=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
}

gpu_a100_40() {
    srun --gpus=a100-40 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
}

gpu_a100_80() {
    srun --gpus=a100-80 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
}

gpu_h100() {
    srun --gpus=h100-96 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
}


# ---------------------------------------------------------------------
# [CLUSTER] Bước 4: Sau khi vào compute node - kiểm tra GPU
# ---------------------------------------------------------------------
# nvidia-smi


# ---------------------------------------------------------------------
# [CLUSTER] Bước 5: Activate conda env & chạy code
# ---------------------------------------------------------------------
# source ~/.bashrc
# conda activate cnn_test
# python3 train_cnn.py


# ---------------------------------------------------------------------
# [CLUSTER] Submit job dạng batch (không cần giữ terminal mở)
# ---------------------------------------------------------------------
# Cần có file run_job.sh dạng:
#
#   #!/bin/bash
#   #SBATCH --job-name=my_run
#   #SBATCH --output=slurm/my_run.out
#   #SBATCH --time=00:30:00
#   #SBATCH --cpus-per-task=4
#   #SBATCH --gpus=a100-40
#   #SBATCH --mem=16GB
#
#   source ~/.bashrc
#   conda activate cnn_test
#   python3 train_cnn.py
#
# rồi chạy:
#   sbatch run_job.sh


# ---------------------------------------------------------------------
# [CLUSTER] Các lệnh theo dõi/quản lý job hay dùng
# ---------------------------------------------------------------------
myjobs() {
    squeue -u "$USER"
}

canceljob() {
    # dùng: canceljob <job_id>
    scancel "$1"
}

joblog() {
    # dùng: joblog <ten_job_khong_duoi>
    cat "slurm/$1.out"
}


# ---------------------------------------------------------------------
# [CLUSTER] Đồng bộ code từ Windows lên cluster (không cần Remote-SSH)
# ---------------------------------------------------------------------
# [WINDOWS] scp -o "ProxyJump=bach66@stujump.comp.nus.edu.sg" -r .\local_folder bach66@xlogin.comp.nus.edu.sg:/home/b/bach66/EchoScene/
#
# Hoặc dùng git:
#   [WINDOWS] git add . && git commit -m "update" && git push
#   [CLUSTER] cd ~/EchoScene && git pull