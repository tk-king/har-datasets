import os
import pandas as pd


ACTIVITY_MAP = {
    0: "Unknown",
    1: "No Freeze",
    2: "Freeze",
}


def parse_daphnet(dir: str, activity_id_col: str) -> pd.DataFrame:
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

    # factorize activity_id to start from 0
    df["activity_id"] = pd.factorize(df["activity_id"])[0]
    df["subject_id"] = pd.factorize(df["subject_id"])[0]

    # round all floats to 6 decimal places
    df = df.round(6)

    # reorder columns
    order = ["session_id", "timestamp", "subject_id", "activity_id", "activity_name"]
    df = df[[*order, *df.columns.difference(order, sort=False)]]

    # sort by timestamp for each session
    df = df.sort_values(["session_id", "timestamp"])

    # reset index
    df = df.reset_index(drop=True)

    return df
