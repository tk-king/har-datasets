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
    SplitType,
    SubjCrossValSplit,
    Training,
)

cfg_motion_sense = HARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="motion_sense",
            download_url="https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
            sampling_freq=50,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "downstairs",
                    "upstairs",
                    "walking",
                    "jogging",
                    "sitting",
                    "standing",
                ],
                sensor_channels=[
                    "attitude.roll",
                    "attitude.pitch",
                    "attitude.yaw",
                    "gravity.x",
                    "gravity.y",
                    "gravity.z",
                    "rotationRate.x",
                    "rotationRate.y",
                    "rotationRate.z",
                    "userAcceleration.x",
                    "userAcceleration.y",
                    "userAcceleration.z",
                    "accel_x",
                    "accel_y",
                    "accel_z",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            batch_size=32,
            learning_rate=0.0001,
            num_epochs=100,
            split=Split(
                split_type=SplitType.GIVEN,
                given_split=GivenSplit(
                    train_subj_ids=list(range(1, 7)),
                    val_subj_ids=list(range(7, 9)),
                    test_subj_ids=list(range(9, 10)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                ),
            ),
        ),
    ),
)
