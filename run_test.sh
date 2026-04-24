set -e

# python3 main.py test-infer --test_mode cellpose_model_A
# python3 main.py test-infer --test_mode cellpose_model_B
# python3 main.py test-infer --test_mode cellpose_model_C
# python3 main.py test-infer --test_mode cellpose_model_D

python3 main.py test-score --test_mode cellpose_model_A
python3 main.py test-score --test_mode cellpose_model_B
python3 main.py test-score --test_mode cellpose_model_C
python3 main.py test-score --test_mode cellpose_model_D