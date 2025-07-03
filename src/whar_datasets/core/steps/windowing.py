from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import short_unique_id as suid  # type: ignore


def generate_windowing(
    session_df: pd.DataFrame, session_id: int, window_time: float, overlap: float
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    # create containers for windows and index
    window_dict = defaultdict(list)
    windows: List[pd.DataFrame] = []

    # specify window and stride as timedeltas
    window_timedelta = pd.Timedelta(seconds=window_time)
    stride_timedelta = pd.Timedelta(seconds=window_time * (1 - overlap))

    # specifiy times in session
    start_time = session_df["timestamp"].min()
    end_time = session_df["timestamp"].max()
    current_start_time = start_time

    # generate windows from session
    while current_start_time + window_timedelta <= end_time:
        current_end_time = current_start_time + window_timedelta

        # get mask corresponding to window
        mask = (session_df["timestamp"] >= current_start_time) & (
            session_df["timestamp"] < current_end_time
        )

        # get window based on mask and keep_cols and reset index
        window_df = session_df[mask]
        window_df.reset_index(drop=True)
        windows.append(window_df)

        # add window info to window index
        window_dict["session_id"].append(session_id)
        window_dict["window_id"].append(suid.generate_short_id())

        # step to next window
        current_start_time += stride_timedelta

    # trim to shortest window for batching
    min_len = min([len(window) for window in windows])
    windows = [window[:min_len] for window in windows]

    # create window index
    window_index = pd.DataFrame(window_dict)
    window_index.astype({"window_id": "string"})

    return window_index, windows
