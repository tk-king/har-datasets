from collections import defaultdict
import os
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
    101: "Relaxing",
    102: "time",
    103: "morning",
    104: "Cleanup",
    105: "time",
}

# LL_Left_Arm
LL_LEFT_ARM_MAP = {
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
    406516: "1",  # Open Door 1
    406517: "2",  # Open Door 2
    404516: "1",  # Close Door 1
    404517: "2",  # Close Door 2
    406520: "Fridge",  # Open Fridge
    404520: "Fridge",  # Close Fridge
    406505: "Dishwasher",  # Open Dishwasher
    404505: "Dishwasher",  # Close Dishwasher
    406519: "1",  # Open Drawer 1
    404519: "1",  # Close Drawer 1
    406511: "2",  # Open Drawer 2
    404511: "2",  # Close Drawer 2
    406508: "3",  # Open Drawer 3
    404508: "3",  # Close Drawer 3
    408512: "Table",  # Clean Table
    407521: "Cup",  # Drink from Cup
    405506: "Switch",  # Toggle Switch
}


def parse_opportunity(dir: str, activity_id_col: str) -> pd.DataFrame:
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
