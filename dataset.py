import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path

provided_code = Path('./provided_code')
RESULTS = Path('./results')

def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")


raw = np.fromfile('./FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax', dtype=np.uint16).reshape(-1, 2048, 2048)
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
plt.savefig('dataset_py.png')

cell_boundaries_train_df = pd.read_csv(provided_code / 'cell_boundaries_train.csv')
print("\ncell_boundaries_train.csv")
print(cell_boundaries_train_df.columns)
print(cell_boundaries_train_df.head())
print(cell_boundaries_train_df.shape)
# print(cell_boundaries_train_df[['boundaryX_z0', 'boundaryY_z0']].head())
print(type(cell_boundaries_train_df[['boundaryX_z0', 'boundaryY_z0']].iloc[0, 0]))
print(len(cell_boundaries_train_df[['boundaryX_z0', 'boundaryY_z0']].iloc[0, 0]))
# print(cell_boundaries_train_df['Unnamed: 0'].value_counts())

spots_train_df = pd.read_csv(provided_code / 'spots_train.csv')
print("\nspots_train.csv")
print(spots_train_df.columns)
print(spots_train_df.head())
print(spots_train_df.shape)
print('IMAGE_ROW:')
get_stats(spots_train_df['image_row'])
print('IMAGE_COL:')
get_stats(spots_train_df['image_col'])

test_spots_df = pd.read_csv(provided_code / 'test_spots.csv')
print("\ntest_spots.csv")
print(test_spots_df.columns)
print(test_spots_df.head())
print('GLOBAL_Z:')
get_stats(test_spots_df['global_z'])