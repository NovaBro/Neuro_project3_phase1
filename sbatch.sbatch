#!/bin/bash
#SBATCH --job-name=dir1_cellpose
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --comment=“gpu_mps=yes”
#SBATCH --output=/scratch/ap7151/slurm_logs/dir1_%j.out
#SBATCH --error=/scratch/ap7151/slurm_logs/dir1_%j.err
#SBATCH --signal=B:SIGUSR1@120

singularity exec --nv \
    --overlay /scratch/ap7151/pytorch-2/ray_torch_env.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c “source /ext3/env.sh && cd /scratch/ap7151/project3 && python -u dir1_cellpose_finetune.py” &
wait $![1:22 PM]# SBATCH --requeue

# Need to singulatrity 