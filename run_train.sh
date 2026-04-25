set -e

# === Train Code ===
python3 main.py train > cellpose_train.txt 2> cellpose_train_error.txt
python3 main.py train --custom_data_dir dapi_polyt > dapi_polyt_train.txt 2> dapi_polyt_train_error.txt
python3 main.py train --custom_data_dir average-z-dapi_polyt > average-z_dapi-polyt_train.txt 2> average-z_dapi-polyt_train_error.txt
