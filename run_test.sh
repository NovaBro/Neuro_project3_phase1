set -e

# === Test Inference ===
# Phase 1 Baseline
python3 main.py test-infer --model_type cpsam --format_id dapi-data 
# python3 main.py test-infer --format_id polyt_data
# python3 main.py test-infer --format_id dapi-polyt_75-25
# python3 main.py test-infer --format_id cellpose_model_D

# python3 main.py test-infer --format_id dapi-polyt --model_name my_new_model_epoch_0225
# python3 main.py test-infer --format_id average-z_dapi-polyt --model_name my_new_model_epoch_0120

python3 main.py test-infer --model_type cpsam --format_id dapi-polyt --model_name my_new_model_epoch_cyto3_epoch_0220
# python3 main.py test-infer --model_type cyto3 --format_id dapi-polyt --model_name my_new_model_epoch_cyto3_epoch_0220
# python3 main.py test-infer --model_type cyto3 --format_id average-z_dapi-polyt --model_name my_new_model_epoch_cyto3_epoch_0280

# === Test Scoring ===
# Phase 1 Baseline
python3 main.py test-score --model_type cpsam --format_id dapi-data 
# python3 main.py test-score --format_id polyt_data
# python3 main.py test-score --format_id dapi-polyt_75-25
# python3 main.py test-score --format_id cellpose_model_D

# python3 main.py test-score --format_id dapi-polyt --model_name my_new_model_epoch_0225
# python3 main.py test-score --format_id average-z_dapi-polyt --model_name my_new_model_epoch_0120

python3 main.py test-score --model_type cpsam --format_id dapi-polyt --model_name my_new_model_epoch_cyto3_epoch_0220
# python3 main.py test-score --model_type cyto3 --format_id dapi-polyt --model_name my_new_model_epoch_cyto3_epoch_0220
# python3 main.py test-score --model_type cyto3 --format_id average-z_dapi-polyt --model_name my_new_model_epoch_cyto3_epoch_0280


# WHOLE: DEPRICATED
# ./run_test.sh > ./output_logs/score_1-10.txt 2> ./output_logs/tqdm_1-10.txt
# ./run_test.sh > ./output_logs/score_11-30.txt 2> ./output_logs/tqdm_11-30.txt
# ./run_test.sh > ./output_logs/score_31-40.txt 2> ./output_logs/tqdm_31-40.txt
# Conclusion: For some reason Last FOVs have higher score than first FOVs