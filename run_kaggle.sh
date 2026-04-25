set -e

# === KAGGLE ===
# Inference
# python3 main.py kaggle-infer --test-mode cellpose_model_A
# python3 main.py test-infer --test_mode cellpose_model_B
# python3 main.py test-infer --test_mode cellpose_model_C
# python3 main.py test-infer --test_mode cellpose_model_D

python3 main.py kaggle-infer --test-mode dapi-polyt --model_name my_new_model_epoch_0225
python3 main.py kaggle-infer --test-mode average-z_dapi-polyt --model_name my_new_model_epoch_0120

# results/kaggle/dapi-polyt/my_new_model_epoch_0225
python provided_code/generate_submission.py \
  --mask_A ./results/kaggle/dapi-polyt/my_new_model_epoch_0225/FOV_A_mask.npy \
  --mask_B ./results/kaggle/dapi-polyt/my_new_model_epoch_0225/FOV_B_mask.npy \
  --mask_C ./results/kaggle/dapi-polyt/my_new_model_epoch_0225/FOV_C_mask.npy \
  --mask_D ./results/kaggle/dapi-polyt/my_new_model_epoch_0225/FOV_D_mask.npy \
  --test_spots provided_code/test_spots.csv \
  --output dapi-polyt_full_submission.csv

python provided_code/generate_submission.py \
  --mask_A ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_A_mask.npy \
  --mask_B ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_B_mask.npy \
  --mask_C ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_C_mask.npy \
  --mask_D ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_D_mask.npy \
  --test_spots provided_code/test_spots.csv \
  --output average-z_dapi-polyt_full_submission.csv