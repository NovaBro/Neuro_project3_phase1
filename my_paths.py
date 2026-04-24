import os
from pathlib import Path

provided_code = Path('./provided_code')
# provided_code = Path('/scratch/pl2820/competition/')
# /scratch/pl2820/competition/

RESULTS = Path('./results')
TRAIN = Path('./train')

DATA_DIR = Path('./')

CUSTOM_DATA = Path('./custom_dataset')
CUSTOM_DATA.mkdir(parents=True, exist_ok=True)
COMP_DIR = Path("/scratch/pl2820/competition/")