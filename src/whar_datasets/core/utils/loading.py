import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_windowing(
    windowing_dir: str, windows_dir: str
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    # load window index
    window_index = pd.read_csv(os.path.join(windowing_dir, "window_index.csv"))

    windows: List[pd.DataFrame] = []

    # load windows
    loop = tqdm(range(len(window_index)))
    loop.set_description("Loading windows")

    for i in loop:
        # get window_id
        window_id = window_index.loc[i]["window_id"]
        assert isinstance(window_id, np.integer)

        # load window and append
        window = load_window(windows_dir, int(window_id))
        windows.append(window)

    return window_index, windows


def load_window(windows_dir: str, window_id: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(windows_dir, f"window_{window_id}.csv"))
