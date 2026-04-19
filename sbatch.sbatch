#!/bin/bash
#SBATCH --job-name=wmz2007_neuro_proj
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=012:00:00
#SBATCH --comment=“gpu_mps=yes”
#SBATCH --output=/home/wmz2007/NeuroInformatics/Neuro_project3_phase1/dir1_%j.out
#SBATCH --error=/home/wmz2007/NeuroInformatics/Neuro_project3_phase1/dir1_%j.err
#SBATCH --signal=B:SIGUSR1@120

singularity exec --fakeroot --overlay overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

singularity exec --nv \
    --overlay /scratch/ap7151/pytorch-2/ray_torch_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c “source /ext3/env.sh && cd /scratch/ap7151/project3 && python -u dir1_cellpose_finetune.py” &
wait $![1:22 PM]# SBATCH --requeue

# Need to singulatrity 
#  --nv GPU
# ro
# full path overlay-15GB-500K.ext3
# Save Checkpoints
# in order to use
# https://colab.research.google.com/drive/1O75SCcZnSnFA7zb6xiyxBg_BGHg2HfiE#scrollTo=4KZVJsrDqyO0

mkdir -p ~/singularity_tmp   # or /scratch/wmz2007/singularity_tmp if you prefer
export SINGULARITY_TMPDIR=~/singularity_tmp
export SINGULARITY_CACHEDIR=~/singularity_tmp   # also helps with cache-related prep

singularity exec --fakeroot --overlay overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash