import os
import pandas as pd


def parse_wisdm_19_phone(dir: str) -> pd.DataFrame:
    dir = os.path.join(dir, "wisdm-dataset/wisdm-dataset")

    # required paths
    accel_dir = os.path.join(dir, "raw/phone/", "accel/")
    gyro_dir = os.path.join(dir, "raw/phone/", "gyro/")
    labels_map_path = os.path.join(dir, "activity_key.txt")

    # only keep txt files
    accel_files = sorted([f for f in os.listdir(accel_dir) if f.endswith(".txt")])
    gyro_files = sorted([f for f in os.listdir(gyro_dir) if f.endswith(".txt")])

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

        # Outer merge on timestamp + IDs
        merged = pd.merge(
            accel_df,
            gyro_df,
            on=["timestamp", "subject_id", "activity_id"],
            how="outer",
            sort=True,
        )

        print(merged.head(30))

        # Convert to datetime (adjust unit as needed)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="ns")
        merged.set_index("timestamp", inplace=True)

        # Resample to 20 Hz (50 ms interval)
        resampled = merged.resample("50ms").mean()

        # Interpolate missing sensor values
        resampled = resampled.interpolate(method="linear")

        # Subject and activity IDs are categorical, forward fill them
        resampled["subject_id"] = merged["subject_id"].resample("50ms").ffill()
        resampled["activity_id"] = merged["activity_id"].resample("50ms").ffill()

        dfs.append(resampled)

    df = pd.concat(dfs).reset_index(drop=True)

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

    # convert nanoseconds to seconds
    df["timestamp"] = df["timestamp"] / 1e9

    return df


def parse_wisdm_19_watch(dir: str) -> pd.DataFrame:
    raise NotImplementedError
