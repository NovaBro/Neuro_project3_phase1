# MERFISH Cell Segmentation Project

NYU CS-GY-9223 course project. Pipeline: load .dax images → segment with Cellpose → assign spots to cells → score via ARI.

## Commands

```bash
# Setup environment
bash setup.sh
source ./env/bin/activate

# Run segmentation and generate submission
python main.py --mode submit

# Test against ground truth
python main.py --mode test
```

## Data Formats

- **Training spots**: `spots_train.csv` — uses `barcode_id` (rename to `spot_id`)
- **Test spots**: `test_spots.csv` — uses `spot_id`
- **Mask**: 2048×2048, 0=background, >0=cell ID
- **Submission**: `spot_id, fov, cluster_id` — cluster_id prefixed with FOV (`FOV_001_cell_1`)

## DAX Image Loading

`.dax` files are raw uint16 binary (no header), 30 frames = 6 channels × 5 z-planes:

```python
raw = np.fromfile(filepath, dtype=np.uint16).reshape(-1, 2048, 2048)
# DAPI: frame 6 + z*5   (z=2 → frame 16)
# polyT: frame 5 + z*5  (z=2 → frame 15)
dapi = raw[16]
polyt = raw[15]
```

## Common Pitfalls

- Use `CellposeModel` (v4+), not `models.Cellpose`
- `model.eval()` returns 3 values: masks, flows, styles
- Training data column is `barcode_id`, not `spot_id` — must rename before scoring
- Cluster IDs must include FOV prefix for uniqueness across FOVs
