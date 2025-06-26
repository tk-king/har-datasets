import os
import pandas as pd


def parse_wisdm_12(dir: str, activity_id_col: str) -> pd.DataFrame:
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

    # specify types
    df = df.astype(
        {
            "subject_id": "int32",
            "activity_name": "str",
            "timestamp": "float32",
            "accel_x": "float32",
            "accel_y": "float32",
            "accel_z": "float32",
        }
    )

    # change timestamp to datetime in ns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    # remove rows with missing values
    df = df[(df["accel_x"] != 0) & (df["accel_y"] != 0) & (df["accel_z"] != 0)]

    # add activity_id
    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # convert all to numeric but activity_name
    df = df.astype({"activity_id": "int32", "session_id": "int32"})

    # reset index
    df = df.reset_index(drop=True)

    return df
