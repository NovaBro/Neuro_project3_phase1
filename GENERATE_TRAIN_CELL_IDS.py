import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


spots_train_df = pd.read_csv('spots_train.csv')
spots_train_df = spots_train_df[spots_train_df['fov'] == 'FOV_001']
spots_train_df = spots_train_df[['fov', 'image_row', 'image_col', 'global_x', 'global_y', 'global_z']]

# Get cell boundries and convert from a string to a list
def parse_float_list(text):
    if isinstance(text, str):
        return np.fromstring(text, sep=',').tolist()
    return None

cell_boundaries_train_df = pd.read_csv('cell_boundaries_train.csv')
cell_boundaries_train_df.iloc[:, 1:] = cell_boundaries_train_df.iloc[:, 1:].applymap(parse_float_list)

from tqdm import tqdm
from shapely.geometry import Point, Polygon

# Converts the x and y columns in the cell boundries df and converts them to polygons so point detection
def get_polygon_df(df:pd.DataFrame):
    out_df = df[['Unnamed: 0']]
    for z in range(5):
        z_polygons = []
        for r in range(df.shape[0]):
            if df.iloc[r, 1 + z * 2]:
                # If there is a boundry detected on a z axis
                x_points = df.iloc[r, 1 + z * 2]
                y_points = df.iloc[r, 2 + z * 2]
                points = list(zip(x_points, y_points))
                z_polygons.append(Polygon(points))
            else:
                # If there is a NO boundry detected on a z axis, no polygon
                z_polygons.append(None)

        out_df[f"polygon_z{z}"] = pd.Series(z_polygons)

    return out_df

polygon_df = get_polygon_df(cell_boundaries_train_df)

# WE ITERATE ACROSS ALL SPOTS DETECTED (MASK) FIRST 
submission_df = []
for s in tqdm(range(spots_train_df.shape[0])):
    spot_row = spots_train_df.iloc[s, :]
    global_x = int(spot_row['global_x'])
    global_y = int(spot_row['global_y'])
    global_z = int(spot_row['global_z'])

    submission_row = {
        'spot_id' : s,
        'fov' : spot_row['fov'],
        'gt_cluster_id' : 'background'
    }

    for cell in tqdm(range(polygon_df.shape[0]), leave=False, disable=True):
        cell_row = polygon_df.loc[cell, :]

            # if there is an NA, then the cell is not on this z-plane, skip
        if not cell_row[f"polygon_z{global_z}"]:
            continue

        if (cell_row[f"polygon_z{global_z}"].contains(Point((global_x, global_y) ))):
            submission_row['gt_cluster_id'] = cell_row["Unnamed: 0"]
            break

    submission_df.append(submission_row)

submission_df = pd.DataFrame(submission_df)
