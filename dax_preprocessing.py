"""Load and preprocess MERFISH DAX stacks (polyT / DAPI channel layout)."""

import numpy as np

# Raw stack layout: 6 channels × 5 z-planes, 30 frames total; stride between z is 5.
Z_PLANE_COUNT = 5
FRAME_STRIDE_Z = 5
FRAME_POLY_T_BASE = 5
FRAME_DAPI_BASE = 6


def poly_t_frame(z: int) -> int:
    return FRAME_POLY_T_BASE + z * FRAME_STRIDE_Z


def dapi_frame(z: int) -> int:
    return FRAME_DAPI_BASE + z * FRAME_STRIDE_Z


def load_dax(filepath, height=2048, width=2048):
    """Load a .dax raw image file. Raw uint16 binary, no header."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    n_frames = len(raw) // (height * width)
    return raw.reshape(n_frames, height, width)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def parse_float_list(text):
    """Parse cell-boundary comma-separated floats from CSV."""
    if isinstance(text, str):
        return np.fromstring(text, sep=",").tolist()
    return None


def clip_norm(x, clip_range=(5, 99)):
    vmin, vmax = np.percentile(x, clip_range)
    x = np.clip(x, vmin, vmax)
    return normalize(x)


def generate_mix(dapi_np, polyt_np, balance=0.5):
    return clip_norm(dapi_np) * balance + (1 - balance) * clip_norm(polyt_np)


def generate_z_stack(idx, epi_stack):
    averaged_image = np.zeros((2048, 2048), dtype=np.float64)
    for z in range(Z_PLANE_COUNT):
        averaged_image += epi_stack[idx + z * FRAME_STRIDE_Z]
    return averaged_image / Z_PLANE_COUNT


def slices_for_z_plane(epi_stack, z_plane: int):
    """Single-z DAPI and polyT images for Cellpose preprocessing."""
    dapi = epi_stack[dapi_frame(z_plane)]
    polyt = epi_stack[poly_t_frame(z_plane)]
    return dapi, polyt
