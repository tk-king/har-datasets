from collections import defaultdict
import os
import pandas as pd

NAMES = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temp",
    "hand_acc_x",
    "hand_acc_y",
    "hand_acc_z",
    "hand_acc_2_x",
    "hand_acc_2_y",
    "hand_acc_2_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_x",
    "hand_orient_y",
    "hand_orient_z",
    "hand_orient_w",
    "chest_temp",
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "chest_acc_2_x",
    "chest_acc_2_y",
    "chest_acc_2_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_x",
    "chest_orient_y",
    "chest_orient_z",
    "chest_orient_w",
    "ankle_temp",
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "ankle_acc_2_x",
    "ankle_acc_2_y",
    "ankle_acc_2_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_x",
    "ankle_orient_y",
    "ankle_orient_z",
    "ankle_orient_w",
]

ACTIVITY_MAP = {
    0: "other",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}


def parse_pamap2(dir: str) -> pd.DataFrame:
    dir = os.path.join(dir, "PAMAP2_Dataset/PAMAP2_Dataset/Protocol/")
    files = [f for f in os.listdir(dir) if f.endswith(".dat")]

    sub_dfs = []

    for file in files:
        # get subject id from filename
        subject_id = file.split(".")[0][-1]

        # read file as df
        sub_df = pd.read_table(
            os.path.join(dir, file), header=None, sep="\\s+", names=NAMES
        )

        # add subject id
        sub_df["subject_id"] = subject_id

        # backfill heartrate
        sub_df["heart_rate"] = sub_df["heart_rate"].bfill()

        # change timestamp to datetime in ns
        sub_df["timestamp"] = pd.to_datetime(sub_df["timestamp"], unit="s")

        # map to types
        types_map = defaultdict(lambda: "float32")
        types_map["activity_id"] = "int32"
        types_map["subject_id"] = "int32"
        types_map["timestamp"] = "datetime64[ns]"
        sub_df = sub_df.astype(types_map)

        # interpolate missing values
        sub_df = sub_df.interpolate(method="linear")

        # append to list
        sub_dfs.append(sub_df)

    # concatenate dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # map activity_id to activity_name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

    # map to types
    types_map = defaultdict(lambda: "float32")
    types_map["activity_name"] = "str"
    types_map["activity_id"] = "int32"
    types_map["subject_id"] = "int32"
    types_map["session_id"] = "int32"
    types_map["timestamp"] = "datetime64[ns]"
    df = df.astype(types_map)

    # reset index
    df = df.reset_index(drop=True)

    return df
