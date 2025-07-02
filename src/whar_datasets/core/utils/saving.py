import os
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm


def save_windowing(
    cfg_hash: str,
    windowing_dir: str,
    windows_dir: str,
    window_index: pd.DataFrame,
    windows: List[pd.DataFrame],
) -> None:
    # create windowing directory if it does not exist
    os.makedirs(windowing_dir, exist_ok=True)
    os.makedirs(windows_dir, exist_ok=True)

    # save config hash
    with open(os.path.join(windowing_dir, "cfg_hash.txt"), "w") as f:
        f.write(cfg_hash)

    # save window index
    window_index.to_csv(os.path.join(windowing_dir, "window_index.csv"), index=False)

    # save windows
    loop = tqdm(range(len(window_index)))
    loop.set_description("Saving windows")

    for i in loop:
        # get window_id
        window_id = window_index.loc[i]["window_id"]
        assert isinstance(window_id, np.integer)

        # get and save window
        window = windows[window_id]
        window.to_csv(os.path.join(windows_dir, f"window_{i}.csv"), index=False)
