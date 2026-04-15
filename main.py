import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")

# ---- Load a raw DAPI image for segmentation ----
# raw = np.fromfile('FOV_001/Epi-750s5-635s5-545s1_001_01.dax', dtype=np.uint16).reshape(-1, 2048, 2048)
# dapi_z2  = raw[16]   # DAPI at middle z-plane
# polyt_z2 = raw[15]   # polyT at middle z-plane
# print(raw.shape)
# print(dapi_z2.shape)
# print(polyt_z2.shape)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(dapi_z2)
# ax[1].imshow(polyt_z2)
# plt.show()

# ---- Load a raw DAPI image for segmentation ----
# raw = np.fromfile('./FOV_001/Epi-750s1-635s1-545s1_001_0.dax', dtype=np.uint16).reshape(-1, 2048, 2048)
# print(raw.shape)
# dapi_z2  = raw[3]   # DAPI at middle z-plane
# print((dapi_z2 > 300).sum())
# # dapi_z2 = dapi_z2 * (dapi_z2 < 300) 
# dapi_z2 = dapi_z2 * (dapi_z2 < 300) + ((dapi_z2 < 50) | (dapi_z2 > 300)) * 100
# # print(dapi_z2.shape)
# # print(dapi_z2)
# # get_stats(dapi_z2)
# # print(((dapi_z2 - np.min(dapi_z2)) / (np.max(dapi_z2) - np.min(dapi_z2))))

# fig, ax = plt.subplots()
# ax.imshow(((dapi_z2 - np.min(dapi_z2)) / (np.max(dapi_z2) - np.min(dapi_z2))))
# get_stats(dapi_z2)
# # plt.hist(dapi_z2.flatten(), bins=100)
# plt.show()

# ---- Load a raw DAPI image for segmentation ----
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
plt.show()




# ---- Load decoded spots (training) ----
# spots_train = pd.read_csv('train/ground_truth/spots_train.csv')
# fov column values are strings like 'FOV_001', 'FOV_019', etc.
# ---- Load ground truth cell boundaries (training) ----
# cells_train = pd.read_csv('train/ground_truth/cell_boundaries_train.csv', index_col=0)
# ---- Load training expression matrix ----
# adata = ad.read_h5ad('train/ground_truth/counts_train.h5ad')
# adata.obs['fov'] values are 'FOV_001', 'FOV_002', etc.
# ---- Load FOV metadata to convert pixel <-> µm ----
# meta = pd.read_csv('reference/fov_metadata.csv').set_index('fov')
# fov_x = meta.loc['FOV_001', 'fov_x']             # FOV origin in µm
# fov_y = meta.loc['FOV_001', 'fov_y']
# pixel_size = meta.loc['FOV_001', 'pixel_size']   # 0.109 µm per pixel
# ---- Load test spots (what you assign to cells) ----
# test_spots = pd.read_csv('test_spots.csv')
# fov column values: 'FOV_A', 'FOV_B', 'FOV_C', 'FOV_D'

# ---- Load submission template ----
# sub = pd.read_csv('sample_submission.csv')
