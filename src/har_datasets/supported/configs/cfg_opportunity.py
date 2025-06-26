from har_datasets.config.config import (
    Common,
    Dataset,
    GivenSplit,
    HARConfig,
    Info,
    Preprocessing,
    Selections,
    SlidingWindow,
    Split,
    SubjCrossValSplit,
    Training,
)

cfg_opportunity = HARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="opportunity",
            download_url="https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
            sampling_freq=30,
        ),
        preprocessing=Preprocessing(
            activity_id_col="Locomotion",
            selections=Selections(
                activity_names=["Stand", "Walk", "Sit", "Lie"],
                sensor_channels=[
                    "IMU_BACK_acc_x",
                    "IMU_BACK_acc_y",
                    "IMU_BACK_acc_z",
                    "IMU_BACK_gyro_x",
                    "IMU_BACK_gyro_y",
                    "IMU_BACK_gyro_z",
                    "IMU_BACK_mag_x",
                    "IMU_BACK_mag_y",
                    "IMU_BACK_mag_z",
                    "IMU_BACK_quat_1",
                    "IMU_BACK_quat_2",
                    "IMU_BACK_quat_3",
                    "IMU_BACK_quat_4",
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            batch_size=32,
            learning_rate=0.0001,
            num_epochs=100,
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(1, 3)),
                    val_subj_ids=list(range(3, 4)),
                    test_subj_ids=list(range(4, 5)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[1, 2], [3, 4]],
                ),
            ),
        ),
    ),
)
