import os
from typing import List, Tuple
import pandas as pd


def parse_har_sense(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    files = [f for f in os.listdir(dir) if f.endswith(".csv") and f != "har_sense.csv"]

    sub_dfs = []

    for file in files:
        sub_df = pd.read_csv(os.path.join(dir, file))
        sub_df["subject_id"] = int(file.split("b")[1].split("_")[0])

        # if it has col Axx-Y, rename it to Acc-Y
        if "Axx-Y" in sub_df.columns:
            sub_df = sub_df.rename(columns={"Axx-Y": "Acc-Y"})

        sub_dfs.append(sub_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # rename activity column
    df = df.rename(columns={"activity": "activity_name"})

    # replace downstaires activtiy name with Downstairs, use map
    df["activity_name"] = df["activity_name"].replace(
        {
            "downstaires": "Downstairs",
            "upstaires": "Upstairs",
            "sitting": "Sitting",
            "walking": "Walking",
        }
    )

    # get activity id from activity name
    df["activity_id"] = df["activity_name"].factorize()[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # create session id
    df["session_id"] = changes.cumsum()

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )

    # factorize
    df["activity_id"] = df["activity_id"].factorize()[0]
    df["subject_id"] = df["subject_id"].factorize()[0]
    df["session_id"] = df["session_id"].factorize()[0]

    # create activity index
    activity_index = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .reset_index(drop=True)
    )

    # create session_index
    session_index = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    # create session dfs
    session_dfs = []
    for session_id in session_index["session_id"].unique():
        session_df = df[df["session_id"] == session_id]
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
            ]
        ).reset_index(drop=True)
        session_dfs.append(session_df)

    # set types
    activity_index["activity_id"] = activity_index["activity_id"].astype("int32")
    activity_index["activity_name"] = activity_index["activity_name"].astype("string")
    session_index["session_id"] = session_index["session_id"].astype("int32")
    session_index["subject_id"] = session_index["subject_id"].astype("int32")
    session_index["activity_id"] = session_index["activity_id"].astype("int32")
    for i, session_df in enumerate(session_dfs):
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ms")
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        session_dfs[i] = session_df.round(6)
        session_dfs[i] = session_df.astype(dtypes)

    return activity_index, session_index, session_dfs
