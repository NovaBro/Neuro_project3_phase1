import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path

from my_paths import *

def get_stats(x):
    print(f"MEAN: {np.mean(x)}")
    print(f"MEDIAN: {np.median(x)}")
    print(f"MIN: {np.min(x)}")
    print(f"MAX: {np.max(x)}")


raw = np.fromfile(TRAIN / 'FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax', dtype=np.uint16).reshape(-1, 2048, 2048)
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

cell_boundaries_train_df = pd.read_csv(provided_code / 'train/ground_truth' / 'cell_boundaries_train.csv')
print("\ncell_boundaries_train.csv")
print(cell_boundaries_train_df.columns)
print(cell_boundaries_train_df.head())
print(cell_boundaries_train_df.shape)
# print(cell_boundaries_train_df[['boundaryX_z0', 'boundaryY_z0']].head())
print(type(cell_boundaries_train_df[['boundaryX_z0', 'boundaryY_z0']].iloc[0, 0]))
print(len(cell_boundaries_train_df[['boundaryX_z0', 'boundaryY_z0']].iloc[0, 0]))
# print(cell_boundaries_train_df['Unnamed: 0'].value_counts())

spots_train_df = pd.read_csv(provided_code / 'train/ground_truth' / 'spots_train.csv')
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

train_spots_solution_df = pd.read_csv('./results/spots_train_w_cell_id_solution.csv')
print("\nspots_train_w_cell_id_solution.csv")
print(train_spots_solution_df.columns)
print(train_spots_solution_df.head())
# Various Tests
# print(train_spots_solution_df['fov'].unique())
print(train_spots_solution_df[train_spots_solution_df['fov'] == 'FOV_019']['global_z'].unique())


from matplotlib.widgets import Slider

def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)

fig, ax = plt.subplots()
# ax.imshow(dapi)
epi_stack = load_dax(provided_code / 'train' /  'FOV_001/Epi-750s5-635s5-545s1-473s5-408s5_001.dax')
# provided_code/train/FOV_001
print(f'Epi stack shape: {epi_stack.shape}  (frames, height, width)')
z_plane = 2  # middle z-plane
dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2

# Create slider axis
# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax, 'Amplitude', 0.0, 5.0, valinit=0)

# Update function
def update(val):
    a = slider.val
    z_plane = int(a)  # middle z-plane
    dapi = epi_stack[6 + z_plane * 5]   # frame 16 for z2
    polyt = epi_stack[5 + z_plane * 5]  # frame 15 for z2

    # line.set_ydata(a * np.sin(x))
    ax.imshow(dapi)
    fig.canvas.draw_idle()

# Connect slider to update function
slider.on_changed(update)
plt.show()