from collections import defaultdict
import os
from typing import List, Tuple
import pandas as pd


SENSOR_COLS = [
    "timestamp",  # 1
    "RKN_upper_acc_x",
    "RKN_upper_acc_y",
    "RKN_upper_acc_z",  # 2–4
    "HIP_acc_x",
    "HIP_acc_y",
    "HIP_acc_z",  # 5–7
    "LUA_upper_acc_x",
    "LUA_upper_acc_y",
    "LUA_upper_acc_z",  # 8–10
    "RUA_lower_acc_x",
    "RUA_lower_acc_y",
    "RUA_lower_acc_z",  # 11–13
    "LH_acc_x",
    "LH_acc_y",
    "LH_acc_z",  # 14–16
    "BACK_acc_x",
    "BACK_acc_y",
    "BACK_acc_z",  # 17–19
    "RKN_lower_acc_x",
    "RKN_lower_acc_y",
    "RKN_lower_acc_z",  # 20–22
    "RWR_acc_x",
    "RWR_acc_y",
    "RWR_acc_z",  # 23–25
    "RUA_upper_acc_x",
    "RUA_upper_acc_y",
    "RUA_upper_acc_z",  # 26–28
    "LUA_lower_acc_x",
    "LUA_lower_acc_y",
    "LUA_lower_acc_z",  # 29–31
    "LWR_acc_x",
    "LWR_acc_y",
    "LWR_acc_z",  # 32–34
    "RH_acc_x",
    "RH_acc_y",
    "RH_acc_z",  # 35–37
    # IMU BACK
    "IMU_BACK_acc_x",
    "IMU_BACK_acc_y",
    "IMU_BACK_acc_z",  # 38–40
    "IMU_BACK_gyro_x",
    "IMU_BACK_gyro_y",
    "IMU_BACK_gyro_z",  # 41–43
    "IMU_BACK_mag_x",
    "IMU_BACK_mag_y",
    "IMU_BACK_mag_z",  # 44–46
    "IMU_BACK_quat_1",
    "IMU_BACK_quat_2",
    "IMU_BACK_quat_3",
    "IMU_BACK_quat_4",  # 47–50
    # IMU RUA
    "IMU_RUA_acc_x",
    "IMU_RUA_acc_y",
    "IMU_RUA_acc_z",  # 51–53
    "IMU_RUA_gyro_x",
    "IMU_RUA_gyro_y",
    "IMU_RUA_gyro_z",  # 54–56
    "IMU_RUA_mag_x",
    "IMU_RUA_mag_y",
    "IMU_RUA_mag_z",  # 57–59
    "IMU_RUA_quat_1",
    "IMU_RUA_quat_2",
    "IMU_RUA_quat_3",
    "IMU_RUA_quat_4",  # 60–63
    # IMU RLA
    "IMU_RLA_acc_x",
    "IMU_RLA_acc_y",
    "IMU_RLA_acc_z",  # 64–66
    "IMU_RLA_gyro_x",
    "IMU_RLA_gyro_y",
    "IMU_RLA_gyro_z",  # 67–69
    "IMU_RLA_mag_x",
    "IMU_RLA_mag_y",
    "IMU_RLA_mag_z",  # 70–72
    "IMU_RLA_quat_1",
    "IMU_RLA_quat_2",
    "IMU_RLA_quat_3",
    "IMU_RLA_quat_4",  # 73–76
    # IMU LUA
    "IMU_LUA_acc_x",
    "IMU_LUA_acc_y",
    "IMU_LUA_acc_z",  # 77–79
    "IMU_LUA_gyro_x",
    "IMU_LUA_gyro_y",
    "IMU_LUA_gyro_z",  # 80–82
    "IMU_LUA_mag_x",
    "IMU_LUA_mag_y",
    "IMU_LUA_mag_z",  # 83–85
    "IMU_LUA_quat_1",
    "IMU_LUA_quat_2",
    "IMU_LUA_quat_3",
    "IMU_LUA_quat_4",  # 86–89
    # IMU LLA
    "IMU_LLA_acc_x",
    "IMU_LLA_acc_y",
    "IMU_LLA_acc_z",  # 90–92
    "IMU_LLA_gyro_x",
    "IMU_LLA_gyro_y",
    "IMU_LLA_gyro_z",  # 93–95
    "IMU_LLA_mag_x",
    "IMU_LLA_mag_y",
    "IMU_LLA_mag_z",  # 96–98
    "IMU_LLA_quat_1",
    "IMU_LLA_quat_2",
    "IMU_LLA_quat_3",
    "IMU_LLA_quat_4",  # 99–102
    # IMU L-SHOE
    "IMU_LSHOE_eu_x",
    "IMU_LSHOE_eu_y",
    "IMU_LSHOE_eu_z",  # 103–105
    "IMU_LSHOE_nav_acc_x",
    "IMU_LSHOE_nav_acc_y",
    "IMU_LSHOE_nav_acc_z",  # 106–108
    "IMU_LSHOE_body_acc_x",
    "IMU_LSHOE_body_acc_y",
    "IMU_LSHOE_body_acc_z",  # 109–111
    "IMU_LSHOE_angvel_body_x",
    "IMU_LSHOE_angvel_body_y",
    "IMU_LSHOE_angvel_body_z",  # 112–114
    "IMU_LSHOE_angvel_nav_x",
    "IMU_LSHOE_angvel_nav_y",
    "IMU_LSHOE_angvel_nav_z",  # 115–117
    "IMU_LSHOE_compass",  # 118
    # IMU R-SHOE
    "IMU_RSHOE_eu_x",
    "IMU_RSHOE_eu_y",
    "IMU_RSHOE_eu_z",  # 119–121
    "IMU_RSHOE_nav_acc_x",
    "IMU_RSHOE_nav_acc_y",
    "IMU_RSHOE_nav_acc_z",  # 122–124
    "IMU_RSHOE_body_acc_x",
    "IMU_RSHOE_body_acc_y",
    "IMU_RSHOE_body_acc_z",  # 125–127
    "IMU_RSHOE_angvel_body_x",
    "IMU_RSHOE_angvel_body_y",
    "IMU_RSHOE_angvel_body_z",  # 128–130
    "IMU_RSHOE_angvel_nav_x",
    "IMU_RSHOE_angvel_nav_y",
    "IMU_RSHOE_angvel_nav_z",  # 131–133
    "IMU_RSHOE_compass",  # 134
]

LABEL_COLS = [
    "Locomotion",
    "HL_Activity",
    "LL_Left_Arm",
    "LL_Left_Arm_Object",
    "LL_Right_Arm",
    "LL_Right_Arm_Object",
    "ML_Both_Arms",
]

# Locomotion
LOCOMOTION_MAP = {0: "Unknown", 1: "Stand", 2: "Walk", 4: "Sit", 5: "Lie"}

# HL_Activity
HL_ACTIVITY_MAP = {
    0: "Unknown",
    101: "Relaxing",
    102: "time",
    103: "morning",
    104: "Cleanup",
    105: "time",
}

# LL_Left_Arm
LL_LEFT_ARM_MAP = {
    0: "Unknown",
    201: "unlock",
    202: "stir",
    203: "lock",
    204: "close",
    205: "reach",
    206: "open",
    207: "sip",
    208: "clean",
    209: "bite",
    210: "cut",
    211: "spread",
    212: "release",
    213: "move",
}

# LL_Left_Arm_Object
LL_LEFT_ARM_OBJECT_MAP = {
    0: "Unknown",
    301: "Bottle",
    302: "Salami",
    303: "Bread",
    304: "Sugar",
    305: "Dishwasher",
    306: "Switch",
    307: "Milk",
    308: "lower",  # Drawer3 (lower)
    309: "Spoon",
    310: "cheese",  # Knife cheese
    311: "middle",  # Drawer2 (middle)
    312: "Table",
    313: "Glass",
    314: "Cheese",
    315: "Chair",
    316: "Door1",
    317: "Door2",
    318: "Plate",
    319: "top",  # Drawer1 (top)
    320: "Fridge",
    321: "Cup",
    322: "salami",  # Knife salami
    323: "Lazychair",
}

# LL_Right_Arm
LL_RIGHT_ARM_MAP = {
    0: "Unknown",
    401: "unlock",
    402: "stir",
    403: "lock",
    404: "close",
    405: "reach",
    406: "open",
    407: "sip",
    408: "clean",
    409: "bite",
    410: "cut",
    411: "spread",
    412: "release",
    413: "move",
}

# LL_Right_Arm_Object
LL_RIGHT_ARM_OBJECT_MAP = {
    0: "Unknown",
    501: "Bottle",
    502: "Salami",
    503: "Bread",
    504: "Sugar",
    505: "Dishwasher",
    506: "Switch",
    507: "Milk",
    508: "lower",  # Drawer3 (lower)
    509: "Spoon",
    510: "cheese",  # Knife cheese
    511: "middle",  # Drawer2 (middle)
    512: "Table",
    513: "Glass",
    514: "Cheese",
    515: "Chair",
    516: "Door1",
    517: "Door2",
    518: "Plate",
    519: "top",  # Drawer1 (top)
    520: "Fridge",
    521: "Cup",
    522: "salami",  # Knife salami
    523: "Lazychair",
}

# ML_Both_Arms
ML_BOTH_ARMS_MAP = {
    0: "Unknown",
    406516: "Open Door 1",
    406517: "Open Door 2",
    404516: "Close Door 1",
    404517: "Close Door 2",
    406520: "Open Fridge",
    404520: "Close Fridge",
    406505: "Open Dishwasher",
    404505: "Close Dishwasher",
    406519: "Open Drawer 1",
    404519: "Close Drawer 1",
    406511: "Open Drawer 2",
    404511: "Close Drawer 2",
    406508: "Open Drawer 3",
    404508: "Close Drawer 3",
    408512: "Clean Table",
    407521: "Drink from Cup",
    405506: "Toggle Switch",
}


def parse_opportunity(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    dir = os.path.join(dir, "OpportunityUCIDataset/dataset/")

    files = [file for file in os.listdir(dir) if file.endswith(".dat")]

    sub_dfs = []

    for file in files:
        # get subject id from filename
        subject_id = int(file.split("-")[0][-1])

        # use body sensors and labels
        sub_df = pd.read_table(
            os.path.join(dir, file),
            header=None,
            sep="\\s+",
            usecols=list(range(134)) + list(range(243, 250)),
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

    # specify one label column as activity_id column
    df = df.rename(columns={activity_id_col: "activity_id"})
    df = df.drop(columns=[col for col in LABEL_COLS if col != activity_id_col])

    # identify where activity or subject changes or chnage in nan entries
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # change ms timestamp from ms to ns
    df["timestamp"] = df["timestamp"] * 1e6
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    # map activity_id to activity_name
    match activity_id_col:
        case "Locomotion":
            df["activity_name"] = df["activity_id"].map(LOCOMOTION_MAP)
        case "HL_Activity":
            df["activity_name"] = df["activity_id"].map(HL_ACTIVITY_MAP)
        case "LL_Left_Arm":
            df["activity_name"] = df["activity_id"].map(LL_LEFT_ARM_MAP)
        case "LL_Left_Arm_Object":
            df["activity_name"] = df["activity_id"].map(LL_LEFT_ARM_OBJECT_MAP)
        case "LL_Right_Arm":
            df["activity_name"] = df["activity_id"].map(LL_RIGHT_ARM_MAP)
        case "LL_Right_Arm_Object":
            df["activity_name"] = df["activity_id"].map(LL_RIGHT_ARM_OBJECT_MAP)
        case "ML_Both_Arms":
            df["activity_name"] = df["activity_id"].map(ML_BOTH_ARMS_MAP)
        case _:
            raise ValueError(f"Unknown activity_id_col: {activity_id_col}")

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
