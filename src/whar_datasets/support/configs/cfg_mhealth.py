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

cfg_mhealth = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="mhealth",
            download_url="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
            sampling_freq=50,
            num_of_subjects=10,
            num_of_activities=12,
            num_of_channels=23,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "Standing still",
                    "Sitting and relaxing",
                    "Lying down",
                    "Walking",
                    "Climbing stairs",
                    "Waist bends forward",
                    "Frontal elevation of arms",
                    "Knees bending (crouching)",
                    "Cycling",
                    "Jogging",
                    "Running",
                    "Jump front and back",
                ],
                sensor_channels=[
                    "chest_acc_x",  # acceleration from the chest sensor (X axis)
                    "chest_acc_y",  # acceleration from the chest sensor (Y axis)
                    "chest_acc_z",  # acceleration from the chest sensor (Z axis)
                    "ecg_1",  # electrocardiogram signal (lead 1)
                    "ecg_2",  # electrocardiogram signal (lead 2)
                    "lankle_acc_x",  # acceleration from the left-ankle sensor (X axis)
                    "lankle_acc_y",  # acceleration from the left-ankle sensor (Y axis)
                    "lankle_acc_z",  # acceleration from the left-ankle sensor (Z axis)
                    "lankle_gyro_x",  # gyro from the left-ankle sensor (X axis)
                    "lankle_gyro_y",  # gyro from the left-ankle sensor (Y axis)
                    "lankle_gyro_z",  # gyro from the left-ankle sensor (Z axis)
                    "lankle_mag_x",  # magnetometer from the left-ankle sensor (X axis)
                    "lankle_mag_y",  # magnetometer from the left-ankle sensor (Y axis)
                    "lankle_mag_z",  # magnetometer from the left-ankle sensor (Z axis)
                    "rarm_acc_x",  # acceleration from the right-lower-arm sensor (X axis)
                    "rarm_acc_y",  # acceleration from the right-lower-arm sensor (Y axis)
                    "rarm_acc_z",  # acceleration from the right-lower-arm sensor (Z axis)
                    "rarm_gyro_x",  # gyro from the right-lower-arm sensor (X axis)
                    "rarm_gyro_y",  # gyro from the right-lower-arm sensor (Y axis)
                    "rarm_gyro_z",  # gyro from the right-lower-arm sensor (Z axis)
                    "rarm_mag_x",  # magnetometer from the right-lower-arm sensor (X axis)
                    "rarm_mag_y",  # magnetometer from the right-lower-arm sensor (Y axis)
                    "rarm_mag_z",  # magnetometer from the right-lower-arm sensor (Z axis)
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(1, 7)),
                    val_subj_ids=list(range(7, 9)),
                    test_subj_ids=list(range(9, 11)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                ),
            ),
        ),
    ),
)
