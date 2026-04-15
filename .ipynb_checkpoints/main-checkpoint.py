import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path

def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")

DATA_DIR = Path('/scratch/pl2820/competition/')

# ---- Load a raw DAPI image for segmentation ----
raw = np.fromfile(DATA_DIR / 'train/FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax', dtype=np.uint16).reshape(-1, 2048, 2048)
print(raw.shape)
dapi_z2  = raw[16]   # DAPI at middle z-plane
print((dapi_z2 > 300).sum())
# dapi_z2 = dapi_z2 * ((dapi_z2 < 300) & (dapi_z2 > 300))
# print(dapi_z2.shape)
# print(dapi_z2)
# get_stats(dapi_z2)
# print(((dapi_z2 - np.min(dapi_z2)) / (np.max(dapi_z2) - np.min(dapi_z2))))

fig, ax = plt.subplots()
ax.imshow(((dapi_z2 - np.min(dapi_z2)) / (np.max(dapi_z2) - np.min(dapi_z2))))
get_stats(dapi_z2)
# plt.hist(dapi_z2.flatten(), bins=100)
plt.savefig('test.png')
