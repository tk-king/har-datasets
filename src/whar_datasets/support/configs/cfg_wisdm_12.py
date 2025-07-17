from tqdm import tqdm
from whar_datasets.core.config import WHARConfig

import os
from typing import Dict, Tuple
import pandas as pd


def parse_wisdm_12(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
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

    # remove rows with missing and nan timestamps
    df = df.astype({"timestamp": "float32"})
    df = df[df["timestamp"] != 0]
    df = df[df["timestamp"].notna()]

    # drop nan rows
    df = df.dropna()

    # change timestamp to datetime in ns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    # add activity_id
    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # factorize
    df["activity_id"] = df["activity_id"].factorize()[0]
    df["subject_id"] = df["subject_id"].factorize()[0]
    df["session_id"] = df["session_id"].factorize()[0]

    # create activity index
    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .reset_index(drop=True)
    )

    # create session_metadata
    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    # create sessions
    sessions: Dict[int, pd.DataFrame] = {}

    # loop over sessions
    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        # get session df
        session_df = df[df["session_id"] == session_id]

        # drop metadata cols
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
            ]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ms")
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        session_df = session_df.round(6)
        session_df = session_df.astype(dtypes)

        # add to sessions
        sessions[session_id] = session_df

    # set metadata types
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_wisdm_12 = WHARConfig(
    # Info + common
    dataset_id="wisdm_12",
    download_url="https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
    sampling_freq=20,
    num_of_subjects=36,
    num_of_activities=6,
    num_of_channels=3,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_wisdm_12,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "Walking",
        "Jogging",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing",
    ],
    sensor_channels=[
        "accel_x",
        "accel_y",
        "accel_z",
    ],
    window_time=5,
    window_overlap=0.5,
    # Training (split info)
    given_train_subj_ids=list(range(0, 29)),
    given_test_subj_ids=list(range(29, 36)),
    subj_cross_val_split_groups=[
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35],
    ],
)
