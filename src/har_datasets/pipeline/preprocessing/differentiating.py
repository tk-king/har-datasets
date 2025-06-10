from typing import List
import pandas as pd

BLOCK_COL = "activity_block_id"
EXCLUDE_COLUMNS = ["subj_id", "activity_id", "activity_block_id", "activity_name"]


def differentiate(
    df: pd.DataFrame,
    sampling_rate: float = 50.0,  # Hz, use 1.0 if you want plain difference
    exclude_columns: List[str] = EXCLUDE_COLUMNS,
    block_col: str = BLOCK_COL,
) -> pd.DataFrame:
    # Get sensor columns
    sensor_cols = df.columns.difference(exclude_columns)

    # Compute time step
    dt = 1.0 / sampling_rate

    # Compute derivative
    diff_data = df.groupby(block_col)[sensor_cols].diff() / dt
    diff_data.bfill(inplace=True)

    # Rename columns
    diff_cols = ["diff_" + col for col in sensor_cols]
    diff_data.rename(columns=dict(zip(sensor_cols, diff_cols)), inplace=True)

    # Concatenate and reset index
    data = pd.concat([df[sensor_cols], diff_data, df[exclude_columns]], axis=1)
    data.reset_index(drop=True)

    return data
