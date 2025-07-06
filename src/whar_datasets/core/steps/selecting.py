from typing import List
import pandas as pd


def select_activities(
    session_metadata: pd.DataFrame,
    activity_metadata: pd.DataFrame,
    activity_names: List[str],
) -> pd.DataFrame:
    print("Selecting activities...")

    # get activity ids corresponding to activity names
    activity_ids = activity_metadata[
        activity_metadata["activity_name"].isin(activity_names)
    ]["activity_id"]

    # get session ids corresponding to activiy ids
    session_ids = session_metadata[session_metadata["activity_id"].isin(activity_ids)]

    return session_ids


def select_channels(session_df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    # print("Selecting channels...")

    # if channels is empty, return df
    return session_df[channels + ["timestamp"]] if len(channels) != 0 else session_df
