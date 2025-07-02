from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


def generate_windowing(
    df: pd.DataFrame,
    window_time: float,
    overlap: float,
    exclude_cols: List[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    print("Generating windows...")

    # specify only channel columns for windows
    keep_cols = [col for col in df.columns if col not in exclude_cols]

    # create containers for windows and index
    window_dict: Dict[str, List[int]] = defaultdict(list)
    windows: List[pd.DataFrame] = []

    # specify window and stride as timedeltas
    window_timedelta = pd.Timedelta(seconds=window_time)
    stride_timedelta = pd.Timedelta(seconds=window_time * (1 - overlap))

    loop = tqdm(df["session_id"].unique())
    loop.set_description("Generating windows")

    for session_id in loop:
        # get session-specific dataframe
        session_df = df[df["session_id"] == session_id].copy()

        # get unique subject_id and activity_id of session
        assert session_df["subject_id"].nunique() == 1
        assert session_df["activity_id"].nunique() == 1
        subject_id = session_df["subject_id"].iloc[0]
        activity_id = session_df["activity_id"].iloc[0]

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
            window_df = session_df[mask][keep_cols]
            window_df.reset_index(drop=True)
            windows.append(window_df)

            # add window info to window index
            window_dict["subject_id"].append(subject_id)
            window_dict["activity_id"].append(activity_id)
            window_dict["window_id"].append(len(windows) - 1)

            # step to next window
            current_start_time += stride_timedelta

    # trim to shortest window for batching
    min_len = min([len(window) for window in windows])
    windows = [window[:min_len] for window in windows]

    # create window index
    window_index = pd.DataFrame(window_dict)

    return window_index, windows


# def get_windowing(
#     dataset_dir: str,
#     cfg_hash: str,
#     df: pd.DataFrame,
#     window_time: float,
#     overlap: float,
#     exclude_cols: List[str],
#     normalization: NormType | None,
# ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
#     print("Getting windowing...")

#     windowing_dir = os.path.join(dataset_dir, "windowing/")
#     windows_dir = os.path.join(windowing_dir, "windows/")

#     # check if windowing corresponds to cfg
#     if os.path.exists(windowing_dir) and cfg_hash == load_cfg_hash(windowing_dir):
#         window_index, windows = load_windowing(windowing_dir, windows_dir)

#     # if not, generate and save windowing
#     else:
#         window_index, windows = generate_windowing(
#             df, window_time, overlap, exclude_cols
#         )

#         # apply per sample normalization
#         match normalization:
#             case NormType.STD_PER_SAMPLE:
#                 windows = normalize_per_sample(windows, standardize, exclude_cols)
#             case NormType.MIN_MAX_PER_SAMPLE:
#                 windows = normalize_per_sample(windows, min_max, exclude_cols)

#         save_windowing(cfg_hash, windowing_dir, windows_dir, window_index, windows)

#     return window_index, windows
