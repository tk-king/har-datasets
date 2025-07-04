import re
import os
from typing import List, Tuple
import pandas as pd

SENSOR_COLS = [
    "chest_acc_x",  # acceleration from the chest sensor (X axis)
    "chest_acc_y",  # acceleration from the chest sensor (Y axis)
    "chest_acc_z",  # acceleration from the chest sensor (Z axis)
    "ecg_1",  # electrocardiogram signal (lead 1)
    "ecg_2",  # electrocardiogram signal (lead 2)
    "lankle_acc_x",  # acceleration from the left-ankle sensor (X axis)
    "lankle_acc_y",  # acceleration from the left-ankle sensor (Y axis)
    "lankle_acc_z",  # acceleration from the left-ankle sensor (Z axis)
    "lankle_gyro_x",  # gyro from the left-ankle sensor (X axis)
    "lankle_gyro_y",  # gyro from the left-ankle sensor (Y axis)
    "lankle_gyro_z",  # gyro from the left-ankle sensor (Z axis)
    "lankle_mag_x",  # magnetometer from the left-ankle sensor (X axis)
    "lankle_mag_y",  # magnetometer from the left-ankle sensor (Y axis)
    "lankle_mag_z",  # magnetometer from the left-ankle sensor (Z axis)
    "rarm_acc_x",  # acceleration from the right-lower-arm sensor (X axis)
    "rarm_acc_y",  # acceleration from the right-lower-arm sensor (Y axis)
    "rarm_acc_z",  # acceleration from the right-lower-arm sensor (Z axis)
    "rarm_gyro_x",  # gyro from the right-lower-arm sensor (X axis)
    "rarm_gyro_y",  # gyro from the right-lower-arm sensor (Y axis)
    "rarm_gyro_z",  # gyro from the right-lower-arm sensor (Z axis)
    "rarm_mag_x",  # magnetometer from the right-lower-arm sensor (X axis)
    "rarm_mag_y",  # magnetometer from the right-lower-arm sensor (Y axis)
    "rarm_mag_z",  # magnetometer from the right-lower-arm sensor (Z axis)
]

LABEL_COLS = [
    "activity_id",  # Label (0 for the null class)
]

ACTIVITY_MAP = {
    0: "Unknown",
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front and back",
}


def parse_mhealth(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    dir = os.path.join(dir, "MHEALTHDATASET/")

    files = [file for file in os.listdir(dir) if file.endswith(".log")]

    sub_dfs = []

    for file in files:
        # get subject id from filename
        match = re.search(r"\d+", file)
        assert match
        subject_id = int(match.group(0))

        # use body sensors and labels
        sub_df = pd.read_table(
            os.path.join(dir, file),
            header=None,
            sep="\\s+",
            names=[*SENSOR_COLS, *LABEL_COLS],
        )

        # fill nans in sensor cols with linear interpolation
        sub_df.loc[:, SENSOR_COLS] = sub_df[SENSOR_COLS].interpolate(method="linear")

        # # fills nans in label cols with backward fill
        sub_df.loc[:, LABEL_COLS] = sub_df[LABEL_COLS].bfill()

        # add subject id
        sub_df["subject_id"] = subject_id

        # append to list
        sub_dfs.append(sub_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes or chnage in nan entries
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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
