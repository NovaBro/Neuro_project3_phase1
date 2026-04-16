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
