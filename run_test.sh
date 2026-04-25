set -e

# === Test Inference ===
python3 main.py test-infer --test-mode cellpose_model_A 
# python3 main.py test-infer --test_mode cellpose_model_B
# python3 main.py test-infer --test_mode cellpose_model_C
# python3 main.py test-infer --test_mode cellpose_model_D

python3 main.py test-infer --test-mode dapi-polyt --model_name my_new_model_epoch_0225
python3 main.py test-infer --test-mode average-z_dapi-polyt --model_name my_new_model_epoch_0120

# === Test Scoring ===
python3 main.py test-score --test-mode cellpose_model_A 
# python3 main.py test-score --test_mode cellpose_model_B
# python3 main.py test-score --test_mode cellpose_model_C
# python3 main.py test-score --test_mode cellpose_model_D

python3 main.py test-score --test-mode dapi-polyt --model_name my_new_model_epoch_0225
python3 main.py test-score --test-mode average-z_dapi-polyt --model_name my_new_model_epoch_0120


# WHOLE: DEPRICATED
# ./run_test.sh > ./output_logs/score_1-10.txt 2> ./output_logs/tqdm_1-10.txt
# ./run_test.sh > ./output_logs/score_11-30.txt 2> ./output_logs/tqdm_11-30.txt
# ./run_test.sh > ./output_logs/score_31-40.txt 2> ./output_logs/tqdm_31-40.txt
# Conclusion: For some reason Last FOVs have higher score than first FOVs