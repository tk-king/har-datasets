import os
import pandas as pd


def parse_har_sense(dir: str, activity_id_col: str) -> pd.DataFrame:
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    files.remove("har_sense.csv")

    sub_dfs = []

    for file in files:
        sub_df = pd.read_csv(os.path.join(dir, file))
        sub_df["subject_id"] = int(file.split("b")[1].split("_")[0])
        sub_dfs.append(sub_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # rename activity column
    df = df.rename(columns={"activity": "activity_name"})

    # get activity id from activity name
    df["activity_id"] = df["activity_name"].factorize()[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # create session id
    df["session_id"] = changes.cumsum()

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e9  # 50 Hz → 0.02 seconds -> to ns
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )

    # convert timestamp to datetime in ns
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

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
