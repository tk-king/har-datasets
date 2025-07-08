from whar_datasets.core.config import (
    Common,
    Dataset,
    GivenSplit,
    Parsing,
    WHARConfig,
    Info,
    Preprocessing,
    Selections,
    SlidingWindow,
    Split,
    SubjCrossValSplit,
    Training,
)

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

    activity_dirs = [
        d
        for d in os.listdir(dir)
        if d != "windowing" and d != "ku_har.csv" and d != "cache"
    ]

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

            # convert to datetime
            session_df["timestamp_acc"] = pd.to_datetime(
                session_df["timestamp_acc"], unit="s"
            )
            session_df["timestamp_gyro"] = pd.to_datetime(
                session_df["timestamp_gyro"], unit="s"
            )

            # select columns
            acc_df = session_df[["timestamp_acc", "acc_x", "acc_y", "acc_z"]]
            gyro_df = session_df[["timestamp_gyro", "gyro_x", "gyro_y", "gyro_z"]]

            # sort by timestamp
            acc_df = acc_df.sort_values("timestamp_acc")
            gyro_df = gyro_df.sort_values("timestamp_gyro")

            # merge_asof to match timestamps
            merged_df = pd.merge_asof(
                acc_df,
                gyro_df,
                left_on="timestamp_acc",
                right_on="timestamp_gyro",
                direction="nearest",
            )

            merged_df = merged_df.drop(columns=["timestamp_gyro"])
            merged_df = merged_df.rename(columns={"timestamp_acc": "timestamp"})

            session_metadata_dict["subject_id"].append(subject_id)
            session_metadata_dict["activity_id"].append(activity_id)

            session_dfs.append(merged_df)

    # define activity index
    activity_metadata = pd.DataFrame(
        list(ACTIVITY_MAP.items()), columns=["activity_id", "activity_name"]
    )

    # define session index
    session_metadata = pd.DataFrame(session_metadata_dict)
    session_metadata["session_id"] = list(range(len(session_dfs)))

    # factorize to start from 0
    session_metadata["activity_id"] = pd.factorize(session_metadata["activity_id"])[0]
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
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="ku_har",
            download_url="https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded",
            sampling_freq=100,
            num_of_subjects=89,
            num_of_activities=18,
            num_of_channels=6,
        ),
        parsing=Parsing(
            parse=parse_ku_har,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
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
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(0, 60)),
                    val_subj_ids=list(range(60, 75)),
                    test_subj_ids=list(range(75, 90)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[
                        list(range(start, start + 10)) for start in range(0, 90)
                    ],
                ),
            ),
        ),
    ),
)
