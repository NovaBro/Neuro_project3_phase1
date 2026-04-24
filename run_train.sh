set -e

# === Train Code ===
python3 main.py train > cellpose_train.txt 2> cellpose_train_error.txt

