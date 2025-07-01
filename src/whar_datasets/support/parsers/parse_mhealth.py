import re
from collections import defaultdict
import os
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


def parse_mhealth(dir: str, activity_id_col: str) -> pd.DataFrame:
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
    sampling_interval = 1 / 50 * 1e9  # 50 Hz → 0.02 seconds -> to ns
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    # add activity name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

    # factorize activity_id to start from 0
    df["activity_id"] = pd.factorize(df["activity_id"])[0]

    # specify types
    types_map = defaultdict(lambda: "float32")
    types_map["activity_name"] = "str"
    types_map["activity_id"] = "int32"
    types_map["subject_id"] = "int32"
    types_map["session_id"] = "int32"
    types_map["timestamp"] = "datetime64[ns]"
    df = df.astype(types_map)

    # round all floats to 6 decimal places
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

    # sort
    df = df.sort_values(["session_id", "timestamp"])

    # reset index
    df = df.reset_index(drop=True)

    return df
