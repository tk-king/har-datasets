import numpy as np
from whar_datasets.core.config import NormType, WHARConfig

from collections import defaultdict
import os
from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm


ACTIVITY_MAP = {
    0: "Stand",
    1: "Sit",
    2: "Talk-sit",
    3: "Talk-stand",
    4: "Stand-sit",
    5: "Lay",
    6: "Lay-stand",
    7: "Pick",
    8: "Jump",
    9: "Push-up",
    10: "Sit-up",
    11: "Walk",
    12: "Walk-backward",
    13: "Walk-circle",
    14: "Run",
    15: "Stair-up",
    16: "Stair-down",
    17: "Table-tennis",
}


def parse_ku_har(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    session_metadata_dict = defaultdict(list)
    session_dfs = []

    activity_dirs = [d for d in os.listdir(dir) if d != "cache"]

    for activity_dir in activity_dirs:
        # get activity from dirname
        activity_id = int(activity_dir.split(".")[0])

        # get activity dir
        activity_dir = os.path.join(dir, activity_dir)
        assert os.path.isdir(activity_dir)

        # go through activity dir
        for file in os.listdir(activity_dir):
            # get subject id from dirname
            subject_id = int(file.split("_")[0])

            # read csv
            session_df = pd.read_csv(
                os.path.join(activity_dir, file),
                names=[
                    "timestamp_acc",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "timestamp_gyro",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                ],
                header=None,
            )

            # remove rows where timestamp is 0
            session_df = session_df[session_df["timestamp_acc"] != 0]
            session_df = session_df[session_df["timestamp_gyro"] != 0]

            # Interpolate gyro to acc timestamps
            for axis in ["x", "y", "z"]:
                # if any is 0, skip
                if (
                    len(session_df["timestamp_acc"]) == 0
                    or len(session_df["timestamp_gyro"]) == 0
                    or len(session_df[f"gyro_{axis}"]) == 0
                ):
                    continue

                session_df[f"gyro_{axis}"] = np.interp(
                    session_df["timestamp_acc"],
                    session_df["timestamp_gyro"],
                    session_df[f"gyro_{axis}"],
                )

            # Optionally convert timestamps to datetime after interpolation
            session_df["timestamp"] = pd.to_datetime(
                session_df["timestamp_acc"], unit="s"
            )
            session_df = session_df.drop(columns=["timestamp_acc", "timestamp_gyro"])

            # Store results
            session_metadata_dict["subject_id"].append(subject_id)
            session_metadata_dict["activity_id"].append(activity_id)

            session_dfs.append(session_df)

    # define activity index
    activity_metadata = pd.DataFrame(
        list(ACTIVITY_MAP.items()), columns=["activity_id", "activity_name"]
    )

    # define session index
    session_metadata = pd.DataFrame(session_metadata_dict)
    session_metadata["session_id"] = list(range(len(session_dfs)))

    # factorize to start from 0
    session_metadata["subject_id"] = pd.factorize(session_metadata["subject_id"])[0]

    # create sessions
    sessions: Dict[int, pd.DataFrame] = {}

    # loop over sessions
    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        # get session df
        session_df = session_dfs[session_id]

        # drop nan rows
        session_df = session_df.dropna()

        # drop index
        session_df.reset_index(drop=True, inplace=True)

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


cfg_ku_har = WHARConfig(
    # Info fields + common
    dataset_id="ku_har",
    download_url="https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded",
    sampling_freq=100,
    num_of_subjects=89,
    num_of_activities=18,
    num_of_channels=6,
    datasets_dir="./datasets",
    # Parsing fields
    parse=parse_ku_har,
    activity_id_col="activity_id",
    # Preprocessing fields (flatten selections + sliding_window)
    activity_names=[
        "Stand",
        "Sit",
        "Talk-sit",
        "Talk-stand",
        "Stand-sit",
        "Lay",
        "Lay-stand",
        "Pick",
        "Jump",
        "Push-up",
        "Sit-up",
        "Walk",
        "Walk-backward",
        "Walk-circle",
        "Run",
        "Stair-up",
        "Stair-down",
        "Table-tennis",
    ],
    sensor_channels=[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ],
    window_time=2.56,
    window_overlap=0.5,
    # Training fields (flattened splits)
    normalization=NormType.ROBUST_SCALE_GLOBALLY,
    given_train_subj_ids=list(range(0, 72)),
    given_test_subj_ids=list(range(72, 90)),
    subj_cross_val_split_groups=[
        [4, 24, 66, 49, 23, 25, 1, 27, 10, 60, 0, 75, 69, 35, 82],
        [2, 18, 72, 88, 20, 78, 7, 79, 36, 13, 26, 59, 73, 37, 16],
        [46, 70, 62, 40, 32, 8, 67, 39, 53, 55, 87, 68, 84, 44, 58],
        [14, 50, 52, 81, 65, 12, 89, 80, 21, 61, 56, 17, 6, 85, 47],
        [48, 28, 33, 43, 5, 83, 38, 34, 19, 86, 15, 11, 29, 9, 3],
        [42, 57, 41, 74, 30, 54, 64, 45, 77, 76, 22, 63, 71, 51, 31],
    ],
)
