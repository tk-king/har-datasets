from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

BLOCK_COL = "activity_block_id"
EXLUDE_COLS = ["subj_id", "activity_id", "activity_block_id", "activity_name"]


def generate_windows(
    df: pd.DataFrame,
    window_size: int,
    displacement: int,
    block_col: str = BLOCK_COL,
    exclude_cols: List[str] = EXLUDE_COLS,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    window_dict: Dict[str, List[int]] = defaultdict(list)

    windows = []

    for block_id in tqdm(df[block_col].unique()):
        block_df = df[df[block_col] == block_id]

        # assert all subj_ids and activity_ids are same in block_df
        assert block_df["subj_id"].nunique() == 1
        assert block_df["activity_id"].nunique() == 1

        subj_id = block_df["subj_id"].unique()[0]
        activity_id = block_df["activity_id"].unique()[0]

        for i in range(0, block_df.shape[0] - window_size + 1, displacement):
            window_df = block_df.iloc[i : i + window_size]
            window_df = window_df.drop(columns=exclude_cols)
            windows.append(window_df)

            window_dict["subj_id"].append(subj_id)
            window_dict["activity_id"].append(activity_id)
            window_dict["window_id"].append(len(windows) - 1)

    window_index = pd.DataFrame(window_dict)

    return window_index, windows
