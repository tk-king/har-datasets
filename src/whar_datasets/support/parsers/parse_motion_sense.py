from typing import List, Tuple
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

            # get subject id from filename between _ and . but multiple
            subject_id = file.split("_")[1].split(".")[0]

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


def parse_motion_sense(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
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

    # identify where activity or subject changes or change in nan entries
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
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
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
