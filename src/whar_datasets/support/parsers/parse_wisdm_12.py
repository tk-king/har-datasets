import os
from typing import List, Tuple
import pandas as pd


def parse_wisdm_12(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    dir = os.path.join(dir, "WISDM_ar_v1.1/")
    file_path = os.path.join(dir, "WISDM_ar_v1.1_raw.txt")

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse all entries into a list of lists
    data = []

    for line in lines:
        # Remove whitespace and newline characters
        line = line.strip()

        if not line:
            continue

        # Split by semicolon to get individual entries
        entries = line.split(";")

        for entry in entries:
            # Skip empty entries
            if len(entry) == 0:
                continue
            # Some entries have a trailing comma
            if entry[-1] == ",":
                entry = entry[:-1]

            # Split each entry by comma
            fields = entry.split(",")

            # Skip entries with too many or too few entries
            if len(fields) != 6:
                continue

            data.append(fields)

    # Create a DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "subject_id",
            "activity_name",
            "timestamp",
            "accel_x",
            "accel_y",
            "accel_z",
        ],
    )

    # remove rows with missing timestamps
    df = df[df["timestamp"] != 0]

    # add activity_id
    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    # identify where activity or subject changes or timestamp is 0
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # change timestamp to datetime in ns
    df = df.astype({"timestamp": "float32"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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
