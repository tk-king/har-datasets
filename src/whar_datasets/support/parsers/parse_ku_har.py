from collections import defaultdict
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import short_unique_id as suid  # type: ignore

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
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    session_index_dict = defaultdict(list)
    session_dfs = []

    activity_dirs = [d for d in os.listdir(dir) if d != "cache"]

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

            session_index_dict["subject_id"].append(subject_id)
            session_index_dict["activity_id"].append(activity_id)

            session_dfs.append(merged_df)

    # define session index
    session_index = pd.DataFrame(session_index_dict)
    session_index["session_id"] = list(range(len(session_dfs)))
    session_index = session_index.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    # define activity index
    activity_index = pd.DataFrame(
        list(ACTIVITY_MAP.items()), columns=["activity_id", "activity_name"]
    )
    activity_index = activity_index.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )

    # factorize to start from 0
    session_index["activity_id"] = pd.factorize(session_index["activity_id"])[0]
    session_index["subject_id"] = pd.factorize(session_index["subject_id"])[0]

    # round all floats to 6 decimal places
    session_dfs = [session_df.round(6) for session_df in session_dfs]

    print(session_index.head())
    print(activity_index.head())
    print(session_dfs[0].head())

    return activity_index, session_index, session_dfs


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
