import os
from pathlib import Path

provided_code = Path('./provided_code')
RESULTS = Path('./results')
TRAIN = Path('./train')

DATA_DIR = Path('./')
# /scratch/pl2820/competition/

CUSTOM_DATA = Path('./custom_dataset')
CUSTOM_DATA.mkdir(parents=True, exist_ok=True)