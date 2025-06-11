from typing import List
import pandas as pd


def select_subjects(df: pd.DataFrame, subject_ids: List[int]) -> pd.DataFrame:
    # if subject_ids is empty, return df
    if len(subject_ids) == 0:
        return df
    return df[df["subject_id"].isin(subject_ids)]


def select_activities(df: pd.DataFrame, activity_ids: List[int]) -> pd.DataFrame:
    # if activity_ids is empty, return df
    if len(activity_ids) == 0:
        return df
    return df[df["activity_id"].isin(activity_ids)]


def select_channels(df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    # if channels is empty, return df
    if len(channels) == 0:
        return df
    return df.drop(columns=[col for col in df.columns if col not in channels])
