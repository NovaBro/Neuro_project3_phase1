set -e

# === KAGGLE ===
# Inference
python3 main.py kaggle-infer --model_type cyto3 --format_id dapi-data
# python3 main.py test-infer --format_id polyt_data
# python3 main.py test-infer --format_id dapi-polyt_75-25
# python3 main.py test-infer --format_id cellpose_model_D

python3 main.py kaggle-infer --model_type cyto3 --format_id dapi-polyt --model_name my_new_model_cyto3_epoch_0220
# python3 main.py kaggle-infer --model_type cyto3 --format_id average-z_dapi-polyt --model_name my_new_model_cyto3_epoch_0120

# Kaggle Submission Generation
python provided_code/generate_submission.py \
  --mask_A ./results/kaggle/dapi-data/default/FOV_A_mask.npy \
  --mask_B ./results/kaggle/dapi-data/default/FOV_B_mask.npy \
  --mask_C ./results/kaggle/dapi-data/default/FOV_C_mask.npy \
  --mask_D ./results/kaggle/dapi-data/default/FOV_D_mask.npy \
  --test_spots provided_code/test_spots.csv \
  --output dapi_full_submission.csv

python provided_code/generate_submission.py \
  --mask_A ./results/kaggle/dapi-polyt/my_new_model_cyto3_epoch_0220/FOV_A_mask.npy \
  --mask_B ./results/kaggle/dapi-polyt/my_new_model_cyto3_epoch_0220/FOV_B_mask.npy \
  --mask_C ./results/kaggle/dapi-polyt/my_new_model_cyto3_epoch_0220/FOV_C_mask.npy \
  --mask_D ./results/kaggle/dapi-polyt/my_new_model_cyto3_epoch_0220/FOV_D_mask.npy \
  --test_spots provided_code/test_spots.csv \
  --output dapi-polyt_full_submission.csv

# python provided_code/generate_submission.py \
#   --mask_A ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_A_mask.npy \
#   --mask_B ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_B_mask.npy \
#   --mask_C ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_C_mask.npy \
#   --mask_D ./results/kaggle/average-z_dapi-polyt/my_new_model_epoch_0120/FOV_D_mask.npy \
#   --test_spots provided_code/test_spots.csv \
#   --output average-z_dapi-polyt_full_submission.csv