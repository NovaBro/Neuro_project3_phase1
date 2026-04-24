import re
import os
import random
import shutil
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from cellpose.models import CellposeModel
from cellpose import io, models, train
from sklearn.model_selection import train_test_split


from provided_code.metric import score
# from provided_code.generate_submission import build_submission
# from generate_train_submission import build_submission as one_submission
from generate_train_submission_v2 import build_submission

from my_paths import *
# DATA_DIR = '/scratch/vsp7230/Last_Colab/data'
SEED = 42
rng = np.random.default_rng(seed=SEED)

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
    parser.add_argument('--test-mode', 
                        # choices=["cellpose_model_A", "cellpose_model_B", "cellpose_model_C"], 
                        help="Argument to send to test mode")
    parser.add_argument('--batch_size', default=1, 
                        # choices=["cellpose_model_A", "cellpose_model_B", "cellpose_model_C"], 
                        help="batch_size. For cellpose, 8 takes 7GB on gpu")
    return parser.parse_args()

def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min()) 

# Get cell boundries and convert from a string to a list
def parse_float_list(text):
    if isinstance(text, str):
        return np.fromstring(text, sep=',').tolist()
    return None

# Dataset helper functions
def clip_norm(x, clip_range=[5, 99]):
    vmin, vmax = np.percentile(x, clip_range)
    x = np.clip(x, vmin, vmax)
    x = normalize(x)
    return x

# Dataset helper functions
def generate_mix(dapi_np, polyt_np, balance=0.5):
    return clip_norm(dapi_np) * balance + (1-balance) * clip_norm(polyt_np)


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

            weighted_average_input = 0.25 * normalize(polyt) + 0.75 * normalize(dapi)
            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(weighted_average_input, diameter, channels)
            return masks, flows, styles

        case "cellpose_model_D":
            dapi, polyt = input_data

            vmin, vmax = np.percentile(polyt, [2, 98])
            clip_polyt = np.clip(polyt, vmin, vmax)

            weighted_average_input = 0.25 * normalize(clip_polyt) + 0.75 * normalize(dapi)
            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(weighted_average_input, diameter, channels)
            return masks, flows, styles

        case "trained-A":
            dapi, polyt = input_data

            # eval() returns 3 values: masks, flows, styles
            masks, flows, styles = model.eval(generate_mix(dapi, polyt) * 255, diameter, channels)
            return masks, flows, styles





args = get_args()

# === Initial Setup ===
# NOTE: ADD TEST SPLIT FILTER HERE
all_fovs = [fov for fov in os.listdir(TRAIN) if fov.find('FOV_') != -1]
# all_fovs.sort()

train_fovs, test_fovs = train_test_split(all_fovs, train_size=(90/100), random_state=SEED)
train_fovs, val_fovs = train_test_split(train_fovs, train_size=(80/90), random_state=SEED)

custom_data_dir = CUSTOM_DATA / 'dapi_polyt'

# === Test Variables ===
format_id = args.test_mode

# Test different sections of the data
# all_fovs.sort()
# test_fovs = all_fovs
# test_fovs = test_fovs[0:(len(test_fovs) // 4)]
# test_fovs = test_fovs[(len(test_fovs) // 4):(len(test_fovs) // 4) * 3]
# test_fovs = test_fovs[-(len(test_fovs) // 4):]

# fov_files = [fov for fov in os.listdir(TRAIN) if fov.find('FOV_') != -1]
# fov_files = [fov for fov in fov_files if fov in test_fovs]

z_planes = [2]
# z_planes = [0, 1, 2, 3, 4]

if args.mode == 'submit-kaggle':
    print("Uset the sbatch to create kaggle submission")
    pass

elif args.mode == 'test-infer':

    # === Load Model ===
    # NOTE TEST DIFFERENT MODELS HERE
    # Cellpose v4+: use CellposeModel (not models.Cellpose)
    # model = CellposeModel(model_type='nuclei', gpu=True)

    model = CellposeModel(model_type='nuclei', gpu=True, pretrained_model='./models/my_new_model_epoch_0225')

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
    for fov in tqdm(test_fovs, desc="Testing on FOVs"):
        fov_num = fov.split('_')[1]
        epi_stack = load_dax(TRAIN / f'{fov}/Epi-750s5-635s5-545s1-473s5-408s5_{fov_num}.dax')
        # print(f'Epi stack shape: {epi_stack.shape}  (frames, height, width)')
        
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

elif args.mode == 'test-score':
    # 2. Run Test
    # fov_files_inference = [fov for fov in  os.listdir(RESULTS / format_id) if fov.find('FOV_') != -1]
    print("Loading spots_train_w_cell_id_solution.csv ...")
    train_solution_df = pd.read_csv("results/spots_train_w_cell_id_solution.csv")
    train_solution_df = train_solution_df[train_solution_df['fov'].isin(test_fovs)] # NOTE: ADD TEST SPLIT FILTER HERE
    print(f" {len(train_solution_df):,} spots across {train_solution_df['fov'].nunique()} FOVs")

    # NOTE: Testing across each z-levels, since the submission function, 
    # and therefore the score function, does not account for the z-level
    average_score_across_z = 0

    for z_level in z_planes:
        sub_train_solution_df = train_solution_df[train_solution_df['global_z'] == float(z_level)]

        if args.verbose: print(f"\nLoading masks at z level {z_level} ...") 
        masks = {}
        for fov in test_fovs:
            masks[f"{fov}"] = np.load(RESULTS / format_id / f"{fov}_z{z_level}_mask.npy")

        submit_df = build_submission(masks, sub_train_solution_df)
        score_at_z = score(sub_train_solution_df, submit_df, 'spot_id')
        average_score_across_z += score_at_z
        if args.verbose: print(f"  Score on z level {z_level}: {score_at_z}")
    
    print(f"==== Final Results ====")
    print(f"Format: {format_id}")
    print(f"Final Score average_score_across_z: {average_score_across_z / len(z_planes)}\n")

elif args.mode == 'gen-data':
    shutil.rmtree((custom_data_dir / 'train'))
    shutil.rmtree((custom_data_dir / 'val'))
    (custom_data_dir / 'train').mkdir(parents=True, exist_ok=True)
    (custom_data_dir / 'val').mkdir(parents=True, exist_ok=True)

    # Load the spots train solution, with cell ids, only for FOV_001 and at z = 2
    train_solution_df = pd.read_csv('results/spots_train_w_cell_id_solution.csv')

    # Get cell boundries and convert from a string to a list
    cell_boundaries_train_df = pd.read_csv(provided_code / 'train/ground_truth' / 'cell_boundaries_train.csv')
    cell_boundaries_train_df.iloc[:, 1:] = cell_boundaries_train_df.iloc[:, 1:].applymap(parse_float_list)
    cell_boundaries_train_df.rename({'Unnamed: 0':'gt_cluster_id'}, inplace=True, axis=1)

    # Cell Boundry Mapping
    reference_xy = pd.read_csv(provided_code / 'reference' /'fov_metadata.csv')

    for train_val_fov in [(train_fovs, 'train'), (val_fovs, 'val')]:
        for fov in tqdm(train_val_fov[0]):
            fov_num = fov.split('_')[1]

            # Load the spots train solution, with cell ids, only for FOV_*** and at z = 2
            train_solution_df_fov = train_solution_df[(train_solution_df['fov'] == fov) 
                                                & (train_solution_df['global_z'] == 2.0)] # NOTE: ONLY Z = 2
            # Cell Boundry Mapping
            reference_xy_fov = reference_xy[reference_xy['fov'] == fov]

            # Only get cell boundries found in FOV_*** and z = 2
            solution_cells_df = cell_boundaries_train_df.merge(train_solution_df_fov, how='inner', on='gt_cluster_id')
            solution_cells_df = solution_cells_df[['gt_cluster_id', 'boundaryX_z2','boundaryY_z2' , 'image_row','image_col', 'fov', 'spot_id']]

            # Consolidate so each cell gets one row and 1 boundry
            def agg_func(df):
                df = df.sort_values(['image_row', 'image_col'])
                x_b = df['boundaryX_z2'].iloc[0]
                y_b = df['boundaryY_z2'].iloc[0]

                return pd.Series({
                    'boundaryX_z2' : x_b,
                    'boundaryY_z2' : y_b,
                })

            apply_solution_cells_df = solution_cells_df.groupby('gt_cluster_id').apply(agg_func)
            apply_solution_cells_df = apply_solution_cells_df.reset_index()

            # --- 1. Initialize the master mask OUTSIDE the loop ---
            # Using int32 to accommodate many cell IDs
            master_mask = np.zeros((2048, 2048), dtype=np.int32)

            for i in range(len(apply_solution_cells_df)):
                x_data = apply_solution_cells_df.loc[i, 'boundaryX_z2']
                y_data = apply_solution_cells_df.loc[i, 'boundaryY_z2']

                # Your coordinate transformation math
                x_px = 2048 - (np.array(x_data) - np.array(reference_xy_fov['fov_x'])) / np.array(reference_xy_fov['pixel_size'])
                y_px = (np.array(y_data) - np.array(reference_xy_fov['fov_y'])) / np.array(reference_xy_fov['pixel_size'])
                
                # OpenCV expects (x, y) pairs. 
                # If your intentional swap means X_coordinate = y_px, then zip(y_px, x_px) is correct.
                polygon_points = np.array(list(zip(y_px, x_px)), dtype=np.int32)

                # --- 2. Fill the master_mask in-place ---
                # We use i + 1 so the first cell isn't 0 (background color)
                cv2.fillPoly(master_mask, [polygon_points], color=int(i + 1))

            # MASKED SOLUTION
            cv2.imwrite(custom_data_dir / train_val_fov[1] / f'cells_{fov_num}_masks.png', master_mask)

            # Mixed Input Image
            epi_stack = load_dax(TRAIN /  f"{fov}/Epi-750s5-635s5-545s1-473s5-408s5_{fov.split('_')[1]}.dax")
            z_plane = 2  # middle z-plane
            dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
            polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2

            # NOTE: Dataset changing done here
            cv2.imwrite(custom_data_dir / train_val_fov[1]  / f'cells_{fov_num}_img.png', generate_mix(dapi, polyt) * 255)

            # Original Images:
            # cv2.imwrite(custom_data_dir / train_val_fov[1]  / 'dapi.png', normalize(dapi) * 255)
            # cv2.imwrite(custom_data_dir / train_val_fov[1]  / 'polyt.png', normalize(polyt) * 255)


elif args.mode == 'train':
    # For running locally on PC
    # nohup python3 main.py train > cellpose_train.txt 2> cellpose_train_error.txt &
    train_dir = (os.getcwd() / custom_data_dir / 'train').__str__() + '/'
    test_dir = (os.getcwd() / custom_data_dir / 'val').__str__() + '/'
    print(train_dir, test_dir)

    io.logger_setup()
    output = io.load_train_test_data(train_dir, test_dir, image_filter="_img",
                                    mask_filter="_masks", look_one_level_down=False)
    images, labels, image_names, test_images, test_labels, image_names_test = output

    model = models.CellposeModel(gpu=True)

    model_path, train_losses, test_losses = train.train_seg(model.net,
                                train_data=images, train_labels=labels,
                                test_data=test_images, test_labels=test_labels,
                                weight_decay=0.1, learning_rate=1e-5,
                                save_every=20, save_each=True,
                                batch_size=args.batch_size,
                                n_epochs=250, model_name="my_new_model")
# 20 min per 100 epoch
# 1 hr per 300 epoch
