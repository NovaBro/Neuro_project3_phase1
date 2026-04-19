"""
Generate a Kaggle submission CSV from your segmentation masks.

The competition evaluates spot-to-cell clustering using the Adjusted Rand Index (ARI).
Your submission is just: for each spot in test_spots.csv, which cluster (cell) does it
belong to?

You provide a segmentation mask (2048 x 2048, 0 = background, >0 = cell ID) for each of
the 4 test FOVs. This script looks up each spot in the mask using the pre-computed
image_row / image_col columns from test_spots.csv.

-----------------------------------------------------------------------------
Example usage in a notebook (recommended):
-----------------------------------------------------------------------------

    import pandas as pd
    from generate_submission import build_submission

    # Run your segmentation pipeline on each test FOV.
    masks = {}
    for fov in ['FOV_A', 'FOV_B', 'FOV_C', 'FOV_D']:
        dapi = load_dapi(f'test/{fov}/...')   # your loading code
        masks[fov] = my_segmentation(dapi)     # (2048, 2048), 0 = bg, >0 = cell

    test_spots = pd.read_csv('test_spots.csv')
    submission = build_submission(masks, test_spots)
    submission.to_csv('submission.csv', index=False)

-----------------------------------------------------------------------------
Example usage from the command line:
-----------------------------------------------------------------------------

    python generate_submission.py \
        --mask_A FOV_A_mask.npy \
        --mask_B FOV_B_mask.npy \
        --mask_C FOV_C_mask.npy \
        --mask_D FOV_D_mask.npy \
        --test_spots test_spots.csv \
        --output submission.csv
"""

import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

output_path = Path("assigned_spots")
output_path.mkdir(exist_ok=True)


def build_submission(masks: dict, test_spots: pd.DataFrame) -> pd.DataFrame:
    """
    Build submission while keeping memory usage low (process FOV by FOV).
    """
    required = ['spot_id', 'fov', 'image_row', 'image_col']
    missing = [c for c in required if c not in test_spots.columns]
    if missing:
        raise ValueError(f"test_spots DataFrame missing columns: {missing}")

    # We'll build the result directly on a copy of the needed columns
    submission = test_spots[['spot_id', 'fov']].copy()
    submission['cluster_id'] = 'background'   # default

    output_path = Path("assigned_spots_temp")
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)   # clean previous run if needed
    output_path.mkdir(exist_ok=True)

    for fov, mask in masks.items():
        fov_spots = test_spots[test_spots['fov'] == fov]
        if fov_spots.empty:
            print(f" WARNING: no spots for {fov}, skipping")
            continue

        if mask.shape != (2048, 2048):
            raise ValueError(f"Mask for {fov} must be shape (2048, 2048), got {mask.shape}")

        rows = fov_spots['image_row'].values
        cols = fov_spots['image_col'].values

        valid = (rows >= 0) & (rows < 2048) & (cols >= 0) & (cols < 2048)

        mask_vals = np.zeros(len(fov_spots), dtype=int)
        if valid.any():
            mask_vals[valid] = mask[rows[valid], cols[valid]]

        # Vectorized cluster_id assignment (much better than the old for-loop)
        cluster_ids = np.full(len(fov_spots), 'background', dtype=object)
        positive = mask_vals > 0
        if positive.any():
            # This is fast and low-memory
            cluster_ids[positive] = np.char.add(
                f"{fov}_cell_", mask_vals[positive].astype(str)
            )

        n_assigned = positive.sum()
        print(
            f" {fov}: {len(fov_spots):,} spots, {int(mask.max())} cells in mask, "
            f"{n_assigned:,} spots assigned ({100 * n_assigned / len(fov_spots):.1f}%)"
        )

        # Optional: still write to parquet per FOV if you want persistence
        df_part = pd.DataFrame({
            'spot_id': fov_spots['spot_id'].values,
            'fov': fov,
            'cluster_id': cluster_ids,
        })
        table = pa.Table.from_pandas(df_part, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=['fov'],
            existing_data_behavior='overwrite_or_ignore'
        )

        # === Critical: Update submission in place for this FOV only ===
        # Use index alignment or merge on spot_id for this small slice
        submission.loc[fov_spots.index, 'cluster_id'] = cluster_ids

    # If you prefer a merge-based update (slightly safer with large data):
    # submission['cluster_id'] = submission['cluster_id'].astype('object')
    # for fov in masks.keys():
    #     part = pd.read_parquet(output_path / f"fov={fov}")
    #     submission = submission.merge(
    #         part[['spot_id', 'cluster_id']], 
    #         on='spot_id', how='left', suffixes=('', '_new')
    #     )
    #     submission['cluster_id'] = submission['cluster_id_new'].fillna(submission['cluster_id'])
    #     submission = submission.drop(columns=['cluster_id_new'])

    return submission[['spot_id', 'fov', 'cluster_id']]

"""
    python generate_train_submission.py \
    --mask_A FOV_001_mask.npy \
    --test_spots spots_train.csv \
    --output submission_FOV_001_mask.csv
"""

def main():
    parser = argparse.ArgumentParser(
        description='Generate Kaggle submission CSV from segmentation masks'
    )
    parser.add_argument('--mask_A', required=True, help='Path to FOV_A .npy mask (2048x2048 int)')
    # parser.add_argument('--mask_B', required=True, help='Path to FOV_B .npy mask (2048x2048 int)')
    # parser.add_argument('--mask_C', required=True, help='Path to FOV_C .npy mask (2048x2048 int)')
    # parser.add_argument('--mask_D', required=True, help='Path to FOV_D .npy mask (2048x2048 int)')
    parser.add_argument('--test_spots', default='test_spots.csv', help='Path to spots_train.csv')
    parser.add_argument('--output', default='submission.csv', help='Output submission CSV path')
    args = parser.parse_args()

    print("Loading test_spots.csv...")
    test_spots = pd.read_csv(args.test_spots)
    test_spots = test_spots.rename({'barcode_id' : 'spot_id'}, axis=1)
    test_spots['spot_id'] = test_spots['spot_id'].apply(lambda x: f"spot_{x}")
    print(f"  {len(test_spots):,} spots across {test_spots['fov'].nunique()} FOVs")

    print("\nLoading masks...")
    masks = {
        'FOV_001': np.load(args.mask_A),
        # 'FOV_B': np.load(args.mask_B),
        # 'FOV_C': np.load(args.mask_C),
        # 'FOV_D': np.load(args.mask_D),
    }
    for fov, mask in masks.items():
        print(f"  {fov}: shape={mask.shape}, dtype={mask.dtype}, {int(mask.max())} cells")

    print("\nBuilding submission...")
    submission = build_submission(masks, test_spots)

    # breakpoint()
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission written to {args.output}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Columns: {submission.columns.tolist()}")
    print(f"  Unique clusters: {submission['cluster_id'].nunique()}")
    print(f"  Background spots: {(submission['cluster_id'] == 'background').sum():,}")
    print("\nUpload this file to Kaggle for scoring.")


if __name__ == '__main__':
    main()
