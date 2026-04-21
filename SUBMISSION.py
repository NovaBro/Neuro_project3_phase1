import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
# import pandas as pd
# import anndata as ad
# import matplotlib.pyplot as plt
from cellpose.models import CellposeModel

from metric import score
from generate_submission import build_submission
from generate_train_submission import build_submission as one_submission

COMP_DIR = Path("/scratch/pl2820/competition/")
FOV_list_dir = os.listdir(COMP_DIR / 'test')

def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")

def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)

for f in FOV_list_dir:
    epi_stack = load_dax(COMP_DIR / f"{f}/Epi-750s5-635s5-545s1-473s5-408s5_001.dax")
    print(f'Epi stack shape: {epi_stack.shape}  (frames, height, width)')
    z_plane = 2  # middle z-plane
    dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
    polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2
    # Model evaluation

    # Cellpose v4+: use CellposeModel (not models.Cellpose)
    model = CellposeModel(model_type='nuclei', gpu=True)
    # eval() returns 3 values: masks, flows, styles
    masks, flows, styles = model.eval(dapi, diameter=30, channels=[0, 0])

    print(f'Segmentation complete!')
    print(f'Mask shape: {masks.shape}')
    print(f'Number of cells found: {masks.max()}')

    # Save masks to file
    np.save(f"{f}_mask.npy", masks)

# !    python generate_submission.py \
#         --mask_A FOV_A_mask.npy \
#         --mask_B FOV_B_mask.npy \
#         --mask_C FOV_C_mask.npy \
#         --mask_D FOV_D_mask.npy \
#         --test_spots test_spots.csv \
#         --output full_submission.csv
