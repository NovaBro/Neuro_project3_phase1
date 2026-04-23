import re
import os
import shutil
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from cellpose.models import CellposeModel

from provided_code.metric import score
# from provided_code.generate_submission import build_submission
# from generate_train_submission import build_submission as one_submission
from generate_train_submission_v2 import build_submission

from my_paths import *
# DATA_DIR = '/scratch/vsp7230/Last_Colab/data'

# TODO: In the future, create main utiles py
def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('mode', help="What to do, train, test, etc")
    parser.add_argument('--test_mode', 
                        # choices=["cellpose_model_A", "cellpose_model_B", "cellpose_model_C"], 
                        help="Argument to send to test mode")
    return parser.parse_args()

def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min()) 

def cellpose_model_test_format(input_data:list, model:CellposeModel, format_id, diameter=30, channels=[0, 0]):
    match format_id:
        case "cellpose_model_A":
            dapi, polyt = input_data

            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(dapi, diameter, channels)
            return masks, flows, styles

        case "cellpose_model_B":
            dapi, polyt = input_data

            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(polyt, diameter, channels)
            return masks, flows, styles

        case "cellpose_model_C":
            dapi, polyt = input_data

            weighted_average_input = 0.75 * normalize(polyt) + 0.25 * normalize(dapi)
            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(weighted_average_input, diameter, channels)
            return masks, flows, styles

        case "cellpose_model_D":
            dapi, polyt = input_data

            vmin, vmax = np.percentile(polyt, [20, 95])
            clip_polyt = np.clip(polyt, vmin, vmax)

            weighted_average_input = 0.25 * normalize(clip_polyt) + 0.75 * normalize(dapi)
            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(weighted_average_input, diameter, channels)
            return masks, flows, styles

args = get_args()

if args.mode == 'submit-kaggle':
    print("Uset the sbatch to create kaggle submission")
    pass

elif args.mode == 'test':
    # NOTE: ADD TEST SPLIT FILTER HERE
    # test_fovs = ['FOV_001']
    test_fovs = os.listdir(TRAIN)
    test_fovs.sort()
    # test_fovs = test_fovs[0:(len(test_fovs) // 4)]

    fov_files = [fov for fov in os.listdir(TRAIN) if fov.find('FOV_') != -1]
    fov_files = [fov for fov in fov_files if fov in test_fovs]

    # === Load Model ===
    # NOTE TEST DIFFERENT MODELS HERE
    # Cellpose v4+: use CellposeModel (not models.Cellpose)
    model = CellposeModel(model_type='nuclei', gpu=True)
    # format_id = 'cellpose_model_A'
    # format_id = 'cellpose_model_B'
    # format_id = 'cellpose_model_C'
    format_id = args.test_mode
    # ==================

    # NOTE: COMMENT / UNCOMMENT DEBUGGING
    if (RESULTS / format_id).exists():
        shutil.rmtree(RESULTS / format_id)
    (RESULTS / format_id).mkdir(parents=True, exist_ok=True)

    # 1. Run Inference
    # NOTE: COMMENT / UNCOMMENT DEBUGGING
    for fov in tqdm(fov_files, desc="Testing on FOVs"):
        fov_num = fov.split('_')[1]
        epi_stack = load_dax(TRAIN / f'{fov}/Epi-750s5-635s5-545s1-473s5-408s5_{fov_num}.dax')
        # print(f'Epi stack shape: {epi_stack.shape}  (frames, height, width)')
        # z_planes = [0, 1, 2, 3, 4]
        z_planes = [2]
        for z_plane in z_planes:
            # z_plane = 2  # middle z-plane
            dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
            polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2
            
            # ==== Model Inference ====
            # NOTE TEST DIFFERENT MODELS HERE
            masks, flows, styles = cellpose_model_test_format([dapi, polyt], model, format_id)
            # ========================= 

            # print(f'Segmentation complete!')
            # print(f'Mask shape: {masks.shape}')
            # print(f'Number of cells found: {masks.max()}')
            tqdm.write(f'FOV: {fov},  Number of cells found: {masks.max()}')

            output_file = RESULTS / format_id / f'{fov}_z{z_plane}_mask.npy'
            np.save(output_file, masks)

    # 2. Run Test
    # fov_files_inference = [fov for fov in  os.listdir(RESULTS / format_id) if fov.find('FOV_') != -1]
    print("Loading spots_train_w_cell_id_solution.csv ...")
    train_solution_df = pd.read_csv("results/spots_train_w_cell_id_solution.csv")
    train_solution_df = train_solution_df[train_solution_df['fov'].isin(test_fovs)] # NOTE: ADD TEST SPLIT FILTER HERE
    print(f" {len(train_solution_df):,} spots across {train_solution_df['fov'].nunique()} FOVs")

    # NOTE: Testing across each z-levels, since the submission function, 
    # and therefore the score function, does not account for the z-level
    average_score_across_z = 0
    # z_planes = [0, 1, 2, 3, 4]
    z_planes = [2]
    for z_level in z_planes:
        sub_train_solution_df = train_solution_df[train_solution_df['global_z'] == float(z_level)]

        print(f"\nLoading masks at z level {z_level} ...")
        masks = {}
        for fov in test_fovs:
            masks[f"{fov}"] = np.load(RESULTS / format_id / f"{fov}_z{z_level}_mask.npy")

        submit_df = build_submission(masks, sub_train_solution_df)
        score_at_z = score(sub_train_solution_df, submit_df, 'spot_id')
        average_score_across_z += score_at_z
        print(f"  Score on z level {z_level}: {score_at_z}")
    
    print(f"==== Final Results ====")
    print(f"Format: {format_id}")
    print(f"Final Score average_score_across_z: {average_score_across_z / 5}")



elif args.mode == 'train':
    pass