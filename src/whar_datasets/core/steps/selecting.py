from typing import List
import pandas as pd


def select_activities(
    session_index: pd.DataFrame, activity_index: pd.DataFrame, activity_names: List[str]
) -> pd.DataFrame:
    # print("Selecting activities...")

    activity_ids = activity_index[activity_index["activity_name"].isin(activity_names)][
        "activity_id"
    ]

    return session_index[session_index["activity_id"].isin(activity_ids)]


def select_channels(session_df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    # print("Selecting channels...")

    # if channels is empty, return df
    return session_df[channels + ["timestamp"]] if len(channels) != 0 else session_df
