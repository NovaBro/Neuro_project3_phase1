import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from cellpose.models import CellposeModel

from provided_code.metric import score
from provided_code.generate_submission import build_submission
from generate_train_submission import build_submission as one_submission
from generate_train_submission_v2 import main

# DATA_DIR = '/scratch/vsp7230/Last_Colab/data'
DATA_DIR = './'
FOV_DIR = os.path.join(DATA_DIR, 'FOV_001')

def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--mode')
    return parser.parse_args()

def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)


args = get_args()

if args.mode == 'submit-kaggle':
    print("Uset the sbatch")
    ""
    pass

elif args.mode == 'test':

    pass

elif args.mode == 'train':
    pass