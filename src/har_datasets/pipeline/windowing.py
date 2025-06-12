from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

EXLUDE_COLS = ["subject_id", "activity_id", "activity_block_id", "activity_name"]


def generate_windows(
    df: pd.DataFrame,
    window_time: float,
    overlap: float,
    exclude_cols: List[str] = EXLUDE_COLS,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    keep_cols = [col for col in df.columns if col not in exclude_cols]

    window_dict: Dict[str, List[int]] = defaultdict(list)
    windows: List[pd.DataFrame] = []

    stride_time = window_time * (1 - overlap)

    for session_id in tqdm(df["session_id"].unique()):
        session_df = df[df["session_id"] == session_id].copy()

        # get unique subject_id and activity_id of session
        assert session_df["subject_id"].nunique() == 1
        assert session_df["activity_id"].nunique() == 1
        subject_id = session_df["subject_id"].iloc[0]
        activity_id = session_df["activity_id"].iloc[0]

        # specifiy times
        start_time = session_df["timestamp"].min()
        end_time = session_df["timestamp"].max()
        current_start_time = start_time

        # generate windows from session
        while current_start_time + window_time <= end_time:
            current_end_time = current_start_time + window_time

            # get mask corresponding to window
            mask = (session_df["timestamp"] >= current_start_time) & (
                session_df["timestamp"] < current_end_time
            )

            # get window based on mask
            window_df = session_df[mask][keep_cols]
            windows.append(window_df)

            # add window info to window index
            window_dict["subject_id"].append(subject_id)
            window_dict["activity_id"].append(activity_id)
            window_dict["window_id"].append(len(windows) - 1)

            current_start_time += stride_time

    window_index = pd.DataFrame(window_dict)
    return window_index, windows


# def generate_windows(
#     df: pd.DataFrame,
#     window_size: int,
#     displacement: int,
#     exclude_cols: List[str] = EXLUDE_COLS,
# ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
#     keep_cols = [col for col in df.columns if col not in exclude_cols]

#     window_dict: Dict[str, List[int]] = defaultdict(list)
#     windows: List[pd.DataFrame] = []

#     for session_id in tqdm(df["session_id"].unique()):
#         session_df = df[df["session_id"] == session_id]

#         # get unique subject_id and activity_id of sessoion
#         assert session_df["subject_id"].nunique() == 1
#         assert session_df["activity_id"].nunique() == 1
#         subject_id = session_df["subject_id"].unique()[0]
#         activity_id = session_df["activity_id"].unique()[0]

#         for i in range(0, session_df.shape[0] - window_size + 1, displacement):
#             window_df = session_df.iloc[i : i + window_size][keep_cols]
#             windows.append(window_df)

#             window_dict["subject_id"].append(subject_id)
#             window_dict["activity_id"].append(activity_id)
#             window_dict["window_id"].append(len(windows) - 1)

#     window_index = pd.DataFrame(window_dict)

#     return window_index, windows
