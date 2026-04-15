"""
Generate a Kaggle submission CSV from your pipeline outputs.

You need to provide:
  1. Your decoded spots CSV — columns: fov, x, y, global_z, target_gene
  2. Your cell boundaries CSV — index=cell_id, columns: fov, center_x, center_y, boundaryX_z0..z4, boundaryY_z0..z4
     (center_x/center_y in µm, boundaries in µm)

This script assigns spots to cells and produces the submission CSV.

Usage:
    python generate_submission.py \
        --spots     my_decoded_spots.csv \
        --cells     my_segmented_cells.csv \
        --output    submission.csv

The output CSV has columns: cell_id, fov, center_x, center_y, <gene_1>, ..., <gene_N>
Upload this CSV to Kaggle for scoring.
"""

import argparse
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from pathlib import Path


def parse_boundary(coord_str):
    if isinstance(coord_str, float):
        return None
    return np.array([float(v) for v in coord_str.split(',')])


def assign_spots_to_cells(spots_df, cells_df):
    """Assign decoded spots to segmented cells using point-in-polygon."""

    # Get all test FOVs
    test_fovs = sorted(cells_df['fov'].unique())
    print(f"Test FOVs: {test_fovs}")

    all_counts = []

    for fov in test_fovs:
        spots_fov = spots_df[spots_df['fov'] == fov]
        cells_fov = cells_df[cells_df['fov'] == fov]

        if spots_fov.empty or cells_fov.empty:
            print(f"  FOV {fov}: no spots or no cells, skipping")
            continue

        # Build polygons per cell per z-plane
        cell_polys = {}
        cell_ids = cells_fov.index.tolist()
        for cell_id in cell_ids:
            row = cells_fov.loc[cell_id]
            for z in range(5):
                bx = parse_boundary(row.get(f'boundaryX_z{z}', float('nan')))
                by = parse_boundary(row.get(f'boundaryY_z{z}', float('nan')))
                if bx is None or by is None:
                    continue
                poly = MplPath(np.column_stack([by, bx]))
                cell_polys[(cell_id, z)] = poly

        # Get all genes
        genes = sorted(spots_fov['target_gene'].unique())
        gene_idx = {g: i for i, g in enumerate(genes)}
        cell_idx = {c: i for i, c in enumerate(cell_ids)}
        counts = np.zeros((len(cell_ids), len(genes)), dtype=int)

        # Assign per z-plane
        for z in range(5):
            spots_z = spots_fov[spots_fov['global_z'] == z].reset_index(drop=True)
            if spots_z.empty:
                continue

            # Test points in µm (same coordinate space as boundaries)
            spot_xy = np.column_stack([spots_z['global_y'].values, spots_z['global_x'].values])

            for cell_id in cell_ids:
                if (cell_id, z) not in cell_polys:
                    continue
                inside = cell_polys[(cell_id, z)].contains_points(spot_xy)
                for gene in spots_z.loc[inside, 'target_gene']:
                    if gene in gene_idx:
                        counts[cell_idx[cell_id], gene_idx[gene]] += 1

        fov_df = pd.DataFrame(counts, index=cell_ids, columns=genes)
        fov_meta = cells_fov.loc[cell_ids, ['fov', 'center_x', 'center_y']]
        fov_result = pd.concat([fov_meta, fov_df], axis=1)
        all_counts.append(fov_result)

        n_assigned = counts.sum()
        print(f"  FOV {fov}: {len(cells_fov)} cells, {len(spots_fov):,} spots, "
              f"{n_assigned:,} assigned ({100*n_assigned/len(spots_fov):.1f}%)")

    submission = pd.concat(all_counts)
    submission = submission.fillna(0).astype({c: int for c in submission.columns if c not in ['fov', 'center_x', 'center_y']})
    submission.index.name = 'cell_id'
    return submission


def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission CSV')
    parser.add_argument('--spots', required=True, help='Path to your decoded spots CSV')
    parser.add_argument('--cells', required=True, help='Path to your segmented cells CSV')
    parser.add_argument('--output', default='submission.csv', help='Output submission CSV')
    args = parser.parse_args()

    print("Loading spots...")
    spots = pd.read_csv(args.spots)
    required_spot_cols = ['fov', 'global_x', 'global_y', 'global_z', 'target_gene']
    missing = [c for c in required_spot_cols if c not in spots.columns]
    if missing:
        raise ValueError(f"Spots CSV missing columns: {missing}")
    print(f"  {len(spots):,} spots, FOVs: {sorted(spots['fov'].unique())}")

    print("Loading cells...")
    cells = pd.read_csv(args.cells, index_col=0)
    required_cell_cols = ['fov', 'center_x', 'center_y']
    missing = [c for c in required_cell_cols if c not in cells.columns]
    if missing:
        raise ValueError(f"Cells CSV missing columns: {missing}")
    print(f"  {len(cells)} cells, FOVs: {sorted(cells['fov'].unique())}")

    print("\nAssigning spots to cells...")
    submission = assign_spots_to_cells(spots, cells)

    submission.to_csv(args.output)
    print(f"\nSubmission written to {args.output}")
    print(f"  {len(submission)} cells × {submission.shape[1]} columns")
    print(f"Upload this file to Kaggle for scoring.")


if __name__ == '__main__':
    main()
