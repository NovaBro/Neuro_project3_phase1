#!/bin/bash
#SBATCH --job-name=wmz2007_neuro_proj
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --comment=“gpu_mps=yes”
#SBATCH --output=/home/wmz2007/NeuroInformatics/Neuro_project3_phase1/dir1_%j.out
#SBATCH --error=/home/wmz2007/NeuroInformatics/Neuro_project3_phase1/dir1_%j.err
#SBATCH --signal=B:SIGUSR1@120
set -e

python3 -m venv temp_env
source temp_env/bin/activate
pip3 install cellpose pandas sklearn

python3 SUBMISSION.py

python generate_submission.py \
  --mask_A FOV_A_mask.npy \
  --mask_B FOV_B_mask.npy \
  --mask_C FOV_C_mask.npy \
  --mask_D FOV_D_mask.npy \
  --test_spots test_spots.csv \
  --output full_submission.csv
