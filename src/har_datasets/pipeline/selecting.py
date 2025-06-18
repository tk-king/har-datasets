from typing import List
import pandas as pd


def select_activities(df: pd.DataFrame, activity_names: List[str]) -> pd.DataFrame:
    print("Selecting activities...")

    # if activity_ids is empty, return df
    return (
        df[df["activity_name"].isin(activity_names)] if len(activity_names) != 0 else df
    )


def select_channels(
    df: pd.DataFrame, channels: List[str], exclude_cols: List[str]
) -> pd.DataFrame:
    print("Selecting channels...")

    # if channels is empty, return df
    return df[channels + exclude_cols] if len(channels) != 0 else df
