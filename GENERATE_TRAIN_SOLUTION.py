"""
This file is to generate the solution to the training dataset.
"""
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from paths import *


spots_train_df = pd.read_csv(provided_code / 'spots_train.csv')
# NOTE: Insert code here to do subset
# spots_train_df = spots_train_df[spots_train_df['fov'] == 'FOV_001']
spots_train_df = spots_train_df[['fov', 'image_row', 'image_col', 'global_x', 'global_y', 'global_z']]

# Get cell boundries and convert from a string to a list
def parse_float_list(text):
    if isinstance(text, str):
        return np.fromstring(text, sep=',').tolist()
    return None

cell_boundaries_train_df = pd.read_csv(provided_code / 'cell_boundaries_train.csv')
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

# Claude Optimization
# https://claude.ai/share/db1c3b53-241d-4d0e-9222-abd4566a4d89
from shapely.strtree import STRtree
from shapely.geometry import Point
from tqdm import tqdm
import pandas as pd
import numpy as np

def assign_spots_to_cells(spots_train_df, polygon_df):
    rows = []
    
    # Pre-compute unique z-planes to avoid repeated string formatting
    # z_planes = spots_train_df['global_z'].unique()
    # print("z_planes", z_planes)
    # return
    
    # Pre-build spatial index per z-plane: {z: (STRtree, list of cell ids)}
    spatial_index = {}
    # for z in z_planes:
    for z in range(0, 5):
        col = f"polygon_z{z}"
        # if col not in polygon_df.columns:
        #     continue
        
        # Only keep rows where polygon is not null/falsy for this z
        valid = polygon_df[polygon_df[col].notna() & polygon_df[col].astype(bool)]
        if valid.empty:
            continue
        
        polygons = valid[col].tolist()
        cell_ids = valid["Unnamed: 0"].tolist()
        tree = STRtree(polygons)
        spatial_index[z] = (tree, polygons, cell_ids)
    
    # Main loop — now O(n log m) instead of O(n*m)
    for s, spot_row in enumerate(tqdm(spots_train_df.itertuples(), total=len(spots_train_df))):
        global_x = int(spot_row.global_x)
        global_y = int(spot_row.global_y)
        global_z = int(spot_row.global_z)
        
        gt_cluster_id = 'background'
        
        if global_z in spatial_index:
            tree, polygons, cell_ids = spatial_index[global_z]
            point = Point(global_x, global_y)
            
            # Query returns indices of candidate polygons (bounding box hit)
            candidates = tree.query(point)
            for idx in candidates:
                if polygons[idx].contains(point):
                    gt_cluster_id = cell_ids[idx]
                    break
        
        rows.append({
            'spot_id': s,
            'fov': spot_row.fov,
            'image_row': spot_row.image_row,
            'image_col': spot_row.image_col,
            'global_x': spot_row.global_x,
            'global_y': spot_row.global_y,
            'global_z': spot_row.global_z,
            'gt_cluster_id': gt_cluster_id,
        })
    
    return pd.DataFrame(rows)

submission_df = assign_spots_to_cells(spots_train_df, polygon_df)
submission_df.to_csv(RESULTS / 'spots_train_w_cell_id_solution.csv')
