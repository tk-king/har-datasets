from typing import List
import pandas as pd


def select_activities(df: pd.DataFrame, activity_names: List[str]) -> pd.DataFrame:
    # if activity_ids is empty, return df
    if len(activity_names) == 0:
        return df
    return df[df["activity_name"].isin(activity_names)]


def select_channels(df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    # if channels is empty, return df
    if len(channels) == 0:
        return df
    return df.drop(columns=[col for col in df.columns if col not in channels])
