from typing import List
import pandas as pd


def select_activities(
    session_df: pd.DataFrame, activity_names: List[str]
) -> pd.DataFrame:
    print("Selecting activities...")

    # if activity_ids is empty, return df
    return (
        session_df[session_df["activity_name"].isin(activity_names)]
        if len(activity_names) != 0
        else session_df
    )


def select_channels(session_df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    print("Selecting channels...")

    # if channels is empty, return df
    return session_df[channels + ["timestamp"]] if len(channels) != 0 else session_df
