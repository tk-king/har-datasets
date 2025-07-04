import os
from typing import List, Tuple
import pandas as pd


ACTIVITY_MAP = {
    0: "Unknown",
    1: "No Freeze",
    2: "Freeze",
}


def parse_daphnet(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    dir = os.path.join(dir, "dataset_fog_release/dataset/")

    files = [f for f in os.listdir(dir) if f.endswith(".txt")]

    sub_dfs = []

    for file in files:
        # read file
        sub_df = pd.read_table(
            os.path.join(dir, file),
            sep="\\s+",
            header=None,
            names=[
                "timestamp",
                "shank_acc_x",
                "shank_acc_y",
                "shank_acc_z",
                "thigh_acc_x",
                "thigh_acc_y",
                "thigh_acc_z",
                "trunk_acc_x",
                "trunk_acc_y",
                "trunk_acc_z",
                "activity_id",
            ],
        )

        # add subject id from filename
        sub_df["subject_id"] = int(file[1:3])

        sub_dfs.append(sub_df)

    global_session_id = 0

    for sub_df in sub_dfs:
        # identify where activity or subject changes
        changes = (sub_df["activity_id"] != sub_df["activity_id"].shift(1)) | (
            sub_df["subject_id"] != sub_df["subject_id"].shift(1)
        )

        # compute local session ids
        local_session_ids = changes.cumsum()

        # offset by global counter
        sub_df["session_id"] = local_session_ids + global_session_id

        # update global counter
        global_session_id = sub_df["session_id"].max() + 1

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # convert timestamp to datetime in ns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

    # add activity name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

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
