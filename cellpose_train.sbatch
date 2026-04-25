#!/bin/bash
#SBATCH --job-name=wmz2007_neuro_proj
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=g2-standard-48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=/home/wmz2007/NeuroInformatics/Neuro_project3_phase1/cellpose_train_%j.out
#SBATCH --error=/home/wmz2007/NeuroInformatics/Neuro_project3_phase1/cellpose_train_%j.err
#SBATCH --signal=B:SIGUSR1@120

# Allowed partitions:
# * interactive
# * n2c48m24 —> CPU only
# l4 = 24 GB
# * g2-standard-12 —> 1 L4 GPU
# * g2-standard-24 —> 2 L4 GPUs
# * g2-standard-48 —> 4 L4 GPUs
# * c12m85-a100-1 —> 1 A100 40GB GPU
# * c24m170-a100-2 —> 2 A100 40GB GPUs
# * n1s8-t4-1 —> 1 T4 GPU

set -e

python3 -m venv temp_env
source temp_env/bin/activate
# pip3 install cellpose pandas scikit-learn
pip3 install numpy pandas matplotlib anndata kaggle cellpose scipy scikit-learn tifffile pyarrow shapely tqdm

# python3 SUBMISSION.py
python3 main.py train --batch_size 4  > cellpose_train.txt 2> cellpose_train_error.txt

# Submission Too Kaggle
python provided_code/generate_submission.py \
  --mask_A ./results/kaggle/my_new_model_epoch_0225/FOV_A_mask.npy \
  --mask_B ./results/kaggle/my_new_model_epoch_0225/FOV_B_mask.npy \
  --mask_C ./results/kaggle/my_new_model_epoch_0225/FOV_C_mask.npy \
  --mask_D ./results/kaggle/my_new_model_epoch_0225/FOV_D_mask.npy \
  --test_spots provided_code/test_spots.csv \
  --output cellpose_trained_full_submission.csv
