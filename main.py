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
    parser.add_argument('--mode', choices=['test', 'submit'])
    return parser.parse_args()

def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)


args = get_args()

if args.mode == 'submit':
    # Load the main Epi file
    epi_path = os.path.join(FOV_DIR, 'Epi-750s5-635s5-545s1-473s5-408s5_001.dax')
    epi_stack = load_dax(epi_path)
    if args.verbose: print(f'Epi stack shape: {epi_stack.shape}  (frames, height, width)')

    # Extract DAPI and polyT from the middle z-plane (z2)
    # DAPI frames: [6, 11, 16, 21, 26] for z0-z4
    # polyT frames: [5, 10, 15, 20, 25] for z0-z4
    z_plane = 2  # middle z-plane
    dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
    polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2

    model = CellposeModel(model_type='nuclei', gpu=True)

    # eval() returns 3 values: masks, flows, styles
    masks, flows, styles = model.eval(dapi, diameter=30, channels=[0, 0])

    print(f'Segmentation complete!')
    print(f'Mask shape: {masks.shape}')
    print(f'Number of cells found: {masks.max()}')
    print(f'Number of cells found: {masks.max()}')
    np.save('FOV_001_mask.npy', masks)

    print(""" 
    We just create the mask npy file.
    Need to run generate_submission.py directly for submission, example:

    python generate_train_submission.py \
    --mask_A FOV_001_mask.npy \
    --test_spots spots_train.csv \
    --output submission_FOV_001_mask.csv
    
    OR

    python generate_submission.py \
    --mask_A FOV_A_mask.npy \
    --mask_B FOV_B_mask.npy \
    --mask_C FOV_C_mask.npy \
    --mask_D FOV_D_mask.npy \
    --test_spots test_spots.csv \
    --output submission.csv
    """)
elif args.mode == 'test':
    submission_df = pd.read_csv('submission_FOV_001_mask.csv')
    solution_df = pd.read_csv('spots_train.csv')
    
    print(submission_df.columns)
    print(submission_df.head())
    print(submission_df['cluster_id'].value_counts()) # Unique values exist

    solution_df['spot_id'] = solution_df.index
    solution_df = solution_df.rename(columns={'barcode_id': 'cluster_id'})
    solution_df = solution_df[['spot_id', 'fov', 'cluster_id']]
    print(f"\n{solution_df.columns}")
    print(solution_df.head())

    # score()

elif args.mode == 'test':
    """
    To use the metric function, need to have it in submission format?
    """
    if args.verbose:
        print('Files in FOV_001:')
        for f in sorted(os.listdir(FOV_DIR)):
            size_mb = os.path.getsize(os.path.join(FOV_DIR, f)) / 1e6
            print(f'  {f}  ({size_mb:.1f} MB)')

    # Load the main Epi file
    epi_path = os.path.join(FOV_DIR, 'Epi-750s5-635s5-545s1-473s5-408s5_001.dax')
    epi_stack = load_dax(epi_path)
    if args.verbose: print(f'Epi stack shape: {epi_stack.shape}  (frames, height, width)')

    # Extract DAPI and polyT from the middle z-plane (z2)
    # DAPI frames: [6, 11, 16, 21, 26] for z0-z4
    # polyT frames: [5, 10, 15, 20, 25] for z0-z4
    z_plane = 2  # middle z-plane
    dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
    polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2

    if args.verbose:
        print(f'DAPI shape: {dapi.shape}, dtype: {dapi.dtype}')
        print(f'DAPI range: [{dapi.min()}, {dapi.max()}]')
        print(f'polyT shape: {polyt.shape}, dtype: {polyt.dtype}')
        print(f'polyT range: [{polyt.min()}, {polyt.max()}]')

    # Cellpose v4+: use CellposeModel (not models.Cellpose)
    model = CellposeModel(model_type='nuclei', gpu=True)

    # eval() returns 3 values: masks, flows, styles
    masks, flows, styles = model.eval(dapi, diameter=30, channels=[0, 0])

    print(f'Segmentation complete!')
    print(f'Mask shape: {masks.shape}')
    print(f'Number of cells found: {masks.max()}')

    spots_train_df = pd.read_csv('spots_train.csv')
    spots_train_df = spots_train_df[spots_train_df['fov'] == 'FOV_001']
    score(spots_train_df)
