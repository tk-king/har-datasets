from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

EXLUDE_COLS = ["subject_id", "activity_id", "activity_block_id", "activity_name"]


def generate_windows(
    df: pd.DataFrame,
    window_size: int,
    displacement: int,
    exclude_cols: List[str] = EXLUDE_COLS,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    keep_cols = [col for col in df.columns if col not in exclude_cols]

    window_dict: Dict[str, List[int]] = defaultdict(list)
    windows: List[pd.DataFrame] = []

    for session_id in tqdm(df["session_id"].unique()):
        session_df = df[df["session_id"] == session_id]

        # get unique subject_id and activity_id of sessoion
        assert session_df["subject_id"].nunique() == 1
        assert session_df["activity_id"].nunique() == 1
        subject_id = session_df["subject_id"].unique()[0]
        activity_id = session_df["activity_id"].unique()[0]

        for i in range(0, session_df.shape[0] - window_size + 1, displacement):
            window_df = session_df.iloc[i : i + window_size][keep_cols]
            windows.append(window_df)

            window_dict["subject_id"].append(subject_id)
            window_dict["activity_id"].append(activity_id)
            window_dict["window_id"].append(len(windows) - 1)

    window_index = pd.DataFrame(window_dict)

    return window_index, windows
