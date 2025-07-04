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

cfg_motion_sense = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="motion_sense",
            download_url="https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
            sampling_freq=50,
            num_of_subjects=24,
            num_of_activities=6,
            num_of_channels=18,
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
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(0, 18)),
                    val_subj_ids=list(range(18, 21)),
                    test_subj_ids=list(range(21, 24)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23],
                    ],
                ),
            ),
        ),
    ),
)
