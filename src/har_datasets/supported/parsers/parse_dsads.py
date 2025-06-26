import re
from collections import defaultdict
import os
import pandas as pd

SENSOR_COLS = [
    # Trunk (T)
    "T_xacc",
    "T_yacc",
    "T_zacc",
    "T_xgyro",
    "T_ygyro",
    "T_zgyro",
    "T_xmag",
    "T_ymag",
    "T_zmag",
    # Right Arm (RA)
    "RA_xacc",
    "RA_yacc",
    "RA_zacc",
    "RA_xgyro",
    "RA_ygyro",
    "RA_zgyro",
    "RA_xmag",
    "RA_ymag",
    "RA_zmag",
    # Left Arm (LA)
    "LA_xacc",
    "LA_yacc",
    "LA_zacc",
    "LA_xgyro",
    "LA_ygyro",
    "LA_zgyro",
    "LA_xmag",
    "LA_ymag",
    "LA_zmag",
    # Right Leg (RL)
    "RL_xacc",
    "RL_yacc",
    "RL_zacc",
    "RL_xgyro",
    "RL_ygyro",
    "RL_zgyro",
    "RL_xmag",
    "RL_ymag",
    "RL_zmag",
    # Left Leg (LL)
    "LL_xacc",
    "LL_yacc",
    "LL_zacc",
    "LL_xgyro",
    "LL_ygyro",
    "LL_zgyro",
    "LL_xmag",
    "LL_ymag",
    "LL_zmag",
]

ACTIVITY_MAP = {
    1: "sitting",
    2: "standing",
    3: "lying on back",
    4: "lying on right side",
    5: "ascending stairs",
    6: "descending stairs",
    7: "standing in an elevator still",
    8: "moving around in an elevator",
    9: "walking in a parking lot",
    10: "walking on treadmill (flat, 4 km/h)",
    11: "walking on treadmill (15° incline, 4 km/h)",
    12: "running on treadmill (8 km/h)",
    13: "exercising on a stepper",
    14: "exercising on a cross trainer",
    15: "cycling on exercise bike (horizontal)",
    16: "cycling on exercise bike (vertical)",
    17: "rowing",
    18: "jumping",
    19: "playing basketball",
}


def parse_dsads(dir: str, activity_id_col: str) -> pd.DataFrame:
    dir = os.path.join(dir, "data/")

    sub_dfs = []

    for activity_dir in os.listdir(dir):
        # get activity from dirname
        activity_id = int(activity_dir[1:])

        activity_dir = os.path.join(dir, activity_dir)

        for subject_dir in os.listdir(activity_dir):
            # get subject id from dirname
            subject_id = int(subject_dir[1:])

            subject_dir = os.path.join(activity_dir, subject_dir)

            sub_df = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(subject_dir, file),
                        header=None,
                        names=SENSOR_COLS,
                    )
                    for file in [
                        f for f in os.listdir(subject_dir) if f.endswith(".txt")
                    ]
                ]
            )

            # fill nans in sensor cols with linear interpolation
            sub_df.loc[:, SENSOR_COLS] = sub_df[SENSOR_COLS].interpolate(
                method="linear"
            )

            sub_df["subject_id"] = subject_id
            sub_df["activity_id"] = activity_id

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
    sampling_interval = 1 / 25 * 1e9  # 25 Hz → 0.02 seconds -> to ns
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
