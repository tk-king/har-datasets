from typing import List
from collections import defaultdict
import os
import pandas as pd

NAMES = [
    "attitude.roll",
    "attitude.pitch",
    "attitude.yaw",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
]

ACTIVITY_MAP = {
    "dws": "downstairs",
    "ups": "upstairs",
    "sit": "sitting",
    "std": "standing",
    "wlk": "walking",
    "jog": "jogging",
}


def get_sub_dfs(dir: str, names: List[str] | None) -> List[pd.DataFrame]:
    sub_dfs: List[pd.DataFrame] = []

    for sub_dir in os.listdir(dir):
        # get activity from filename
        activity_id = sub_dir.split("_")[0]

        sub_dir = os.path.join(dir, sub_dir)

        # go through all csv files
        for file in [f for f in os.listdir(sub_dir) if f.endswith(".csv")]:
            file_path = os.path.join(sub_dir, file)

            # get subject id from filename
            subject_id = int(file.split(".")[0][-1])

            # read file as df
            sub_df = (
                pd.read_csv(file_path, names=names, index_col=0, header=0)
                if names is not None
                else pd.read_csv(file_path, index_col=0, header=0)
            )

            # add subject id and activity id
            sub_df["subject_id"] = subject_id
            sub_df["activity_id"] = activity_id

            # append to list
            sub_dfs.append(sub_df)

    return sub_dfs


def parse_motion_sense(dir: str) -> pd.DataFrame:
    dir = os.path.join(dir, "motion-sense-master/data/")
    motion_dir = os.path.join(dir, "A_DeviceMotion_data/A_DeviceMotion_data/")
    accel_dir = os.path.join(dir, "B_Accelerometer_data/B_Accelerometer_data/")
    gyro_dir = os.path.join(dir, "C_Gyroscope_data/C_Gyroscope_data/")

    # get dfs for each sensor type
    motion_dfs = get_sub_dfs(motion_dir, names=None)
    accel_dfs = get_sub_dfs(accel_dir, names=["accel_x", "accel_y", "accel_z"])
    gyro_dfs = get_sub_dfs(gyro_dir, names=["gyro_x", "gyro_y", "gyro_z"])

    # concatenate dfs
    sub_dfs = [
        pd.concat([m_df, a_df, g_df], axis=1)
        for m_df, a_df, g_df in zip(motion_dfs, accel_dfs, gyro_dfs)
    ]

    # remove duplicate cols
    sub_dfs = [df.loc[:, ~df.columns.duplicated()] for df in sub_dfs]

    # concatenate dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes or chnage in nan entries
    changes = (
        (df["activity_id"] != df["activity_id"].shift(1))
        | (df["subject_id"] != df["subject_id"].shift(1))
        | (df.isnull().any(axis=1) != df.isnull().any(axis=1).shift(1))
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # remove nan rows
    df = df.dropna()

    # map activity_id to activity_name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

    # factorize activity_id
    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e9  # 50 Hz → 0.02 seconds -> to ns
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    # map to types
    types_map = defaultdict(lambda: "float32")
    types_map["activity_name"] = "str"
    types_map["activity_id"] = "int32"
    types_map["subject_id"] = "int32"
    types_map["session_id"] = "int32"
    types_map["timestamp"] = "datetime64[ns]"
    df = df.astype(types_map)

    # round all float to 6 decimal places
    df = df.round(6)

    # reorder columns
    order = ["subject_id", "activity_id", "activity_name", "session_id", "timestamp"]
    df = df[
        [
            *order,
            *df.columns.difference(
                order,
                sort=False,
            ),
        ]
    ]

    # reset index
    df = df.reset_index(drop=True)

    return df
