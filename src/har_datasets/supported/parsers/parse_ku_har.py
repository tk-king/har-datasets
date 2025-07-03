from collections import defaultdict
import os
import pandas as pd

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


def parse_ku_har(dir: str, activity_id_col: str) -> pd.DataFrame:
    sub_dfs = []

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
            sub_df = pd.read_csv(
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
            sub_df["timestamp_acc"] = pd.to_datetime(sub_df["timestamp_acc"], unit="s")
            sub_df["timestamp_gyro"] = pd.to_datetime(
                sub_df["timestamp_gyro"], unit="s"
            )

            # select columns
            acc_df = sub_df[["timestamp_acc", "acc_x", "acc_y", "acc_z"]]
            gyro_df = sub_df[["timestamp_gyro", "gyro_x", "gyro_y", "gyro_z"]]

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

            merged_df["subject_id"] = subject_id
            merged_df["activity_id"] = activity_id

            sub_dfs.append(merged_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # add activity name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

    # factorize activity_id to start from 0
    df["activity_id"] = pd.factorize(df["activity_id"])[0]
    df["subject_id"] = pd.factorize(df["subject_id"])[0]

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


# accel_df = df[["timestamp_acc", "acc_x", "acc_y", "acc_z"]]
# gyro_df = df[["timestamp_gyro", "gyro_x", "gyro_y", "gyro_z"]]

# # sort by timestamp
# accel_df = accel_df.sort_values("timestamp_acc")
# gyro_df = gyro_df.sort_values("timestamp_gyro")

# # merge_asof backward (just before or equal)
# backward_df = pd.merge_asof(
#     accel_df,
#     gyro_df,
#     left_on="timestamp_acc",
#     right_on="timestamp_gyro",
#     direction="backward",
# )
# backward_df = backward_df.rename(
#     columns={
#         "timestamp_gyro": "timestamp_gyro_b",
#         "gyro_x": "gyro_x_b",
#         "gyro_y": "gyro_y_b",
#         "gyro_z": "gyro_z_b",
#     }
# )

# # merge_asof forward (just after or equal)
# forward_df = pd.merge_asof(
#     accel_df,
#     gyro_df,
#     left_on="timestamp_acc",
#     right_on="timestamp_gyro",
#     direction="forward",
# )
# forward_df = forward_df.rename(
#     columns={
#         "timestamp_gyro": "timestamp_gyro_f",
#         "gyro_x": "gyro_x_f",
#         "gyro_y": "gyro_y_f",
#         "gyro_z": "gyro_z_f",
#     }
# )

# combined_df = pd.concat([backward_df, forward_df], axis=1)

# # Linear interpolation
# def lerp(t, t0, t1, v0, v1):
#     # Avoid division by zero if timestamps are equal
#     return v0 + (v1 - v0) * ((t - t0) / (t1 - t0)) if t1 != t0 else v0

# # Apply interpolation per row
# def interpolate_row(row):
#     t = row["timestamp_acc"]
#     t0 = row["timestamp_gyro_b"]
#     t1 = row["timestamp_gyro_f"]
#     return pd.Series(
#         {
#             "gyro_x": lerp(t, t0, t1, row["gyro_x_b"], row["gyro_x_f"]),
#             "gyro_y": lerp(t, t0, t1, row["gyro_y_b"], row["gyro_y_f"]),
#             "gyro_z": lerp(t, t0, t1, row["gyro_z_b"], row["gyro_z_f"]),
#         }
#     )

# interp_df = combined_df.apply(interpolate_row, axis=1)

# # Final merged DataFrame
# sub_df = pd.concat(
#     [accel_df, interp_df, df["subject_id"], df["activity_id"]], axis=1
# )
# sub_df.rename(columns={"timestamp_acc": "timestamp"}, inplace=True)

# print(sub_df.columns)
