from whar_datasets.core.config import (
    Common,
    Dataset,
    GivenSplit,
    WHARConfig,
    Info,
    Preprocessing,
    Selections,
    SlidingWindow,
    Split,
    SubjCrossValSplit,
    Training,
)

cfg_opportunity = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="opportunity",
            download_url="https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
            sampling_freq=30,
            num_of_subjects=4,
            num_of_activities=18,
            num_of_channels=133,
        ),
        preprocessing=Preprocessing(
            activity_id_col="ML_Both_Arms",
            selections=Selections(
                activity_names=[
                    "Unknown",  # 0
                    "Open Door 1",  # 406516
                    "Open Door 2",  # 406517
                    "Close Door 1",  # 404516
                    "Close Door 2",  # 404517
                    "Open Fridge",  # 406520
                    "Close Fridge",  # 404520
                    "Open Dishwasher",  # 406505
                    "Close Dishwasher",  # 404505
                    "Open Drawer 1",  # 406519
                    "Close Drawer 1",  # 404519
                    "Open Drawer 2",  # 406511
                    "Close Drawer 2",  # 404511
                    "Open Drawer 3",  # 406508
                    "Close Drawer 3",  # 404508
                    "Clean Table",  # 408512
                    "Drink from Cup",  # 407521
                    "Toggle Switch",  # 405506
                ],
                sensor_channels=[
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
                    "IMU_RUA_acc_x",
                    "IMU_RUA_acc_y",
                    "IMU_RUA_acc_z",  # 51–53
                    "IMU_RUA_gyro_x",
                    "IMU_RUA_gyro_y",
                    "IMU_RUA_gyro_z",  # 54–56
                    "IMU_RUA_mag_x",
                    "IMU_RUA_mag_y",
                    "IMU_RUA_mag_z",  # 57–59
                    "IMU_RLA_acc_x",
                    "IMU_RLA_acc_y",
                    "IMU_RLA_acc_z",  # 64–66
                    "IMU_RLA_gyro_x",
                    "IMU_RLA_gyro_y",
                    "IMU_RLA_gyro_z",  # 67–69
                    "IMU_RLA_mag_x",
                    "IMU_RLA_mag_y",
                    "IMU_RLA_mag_z",  # 70–72
                    "IMU_LUA_acc_x",
                    "IMU_LUA_acc_y",
                    "IMU_LUA_acc_z",  # 77–79
                    "IMU_LUA_gyro_x",
                    "IMU_LUA_gyro_y",
                    "IMU_LUA_gyro_z",  # 80–82
                    "IMU_LUA_mag_x",
                    "IMU_LUA_mag_y",
                    "IMU_LUA_mag_z",  # 83–85
                    "IMU_LLA_acc_x",
                    "IMU_LLA_acc_y",
                    "IMU_LLA_acc_z",  # 90–92
                    "IMU_LLA_gyro_x",
                    "IMU_LLA_gyro_y",
                    "IMU_LLA_gyro_z",  # 93–95
                    "IMU_LLA_mag_x",
                    "IMU_LLA_mag_y",
                    "IMU_LLA_mag_z",  # 96–98
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
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(0, 2)),
                    val_subj_ids=list(range(2, 3)),
                    test_subj_ids=list(range(3, 4)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[0, 1], [2, 3]],
                ),
            ),
        ),
    ),
)
