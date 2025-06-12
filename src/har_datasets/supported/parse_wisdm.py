import os
import pandas as pd


def parse_wisdm_phone(dir: str) -> pd.DataFrame:
    dir = os.path.join(dir, "wisdm-dataset/wisdm-dataset")

    accel_dir = os.path.join(dir, "raw/phone/", "accel/")
    gyro_dir = os.path.join(dir, "raw/phone/", "gyro/")
    labels_map_path = os.path.join(dir, "activity_key.txt")

    # only keep txt files
    accel_files = os.listdir(accel_dir)
    gyro_files = os.listdir(gyro_dir)
    accel_files = [f for f in accel_files if f.endswith(".txt")]
    gyro_files = [f for f in gyro_files if f.endswith(".txt")]
    accel_files = sorted(accel_files)
    gyro_files = sorted(gyro_files)

    dfs = []

    for accel_file, gyro_file in zip(accel_files, gyro_files):
        accel_df = pd.read_csv(
            os.path.join(accel_dir, accel_file),
            header=None,
            names=[
                "subject_id",
                "activity_id",
                "timestamp",
                "accel_x",
                "accel_y",
                "accel_z",
            ],
        )

        gyro_df = pd.read_csv(
            os.path.join(gyro_dir, gyro_file),
            header=None,
            names=[
                "subject_id",
                "activity_id",
                "timestamp",
                "gyro_x",
                "gyro_y",
                "gyro_z",
            ],
        )

        accel_df.sort_values(by=["timestamp"], inplace=True)
        gyro_df.sort_values(by=["timestamp"], inplace=True)

        # Merge using nearest timestamp within matching subject and activity
        merged = pd.merge_asof(
            accel_df,
            gyro_df,
            on="timestamp",
            by=["subject_id", "activity_id"],
            direction="nearest",  # closest timestamp
            tolerance=1000000,  # adjust based on max acceptable time difference in nanoseconds
        )

        merged.dropna(inplace=True)

        dfs.append(merged)

    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    # add col with activity names
    label_dict = {}
    with open(labels_map_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                label_dict[value.strip()] = key.strip()

    df["activity_name"] = df["activity_id"].map(label_dict)

    # convert activity_id to categorical starting from 0
    df["activity_id"] = pd.factorize(df["activity_id"])[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    return df


def parse_wisdm_watch(dir: str) -> pd.DataFrame:
    raise NotImplementedError
