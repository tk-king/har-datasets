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

cfg_pamap2 = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="pamap2",
            download_url="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
            sampling_freq=100,
            num_of_subjects=9,
            num_of_activities=13,
            num_of_channels=52,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "other",
                    "lying",
                    "sitting",
                    "standing",
                    "ironing",
                    "vacuum cleaning",
                    "ascending stairs",
                    "descending stairs",
                    "walking",
                    "cycling",
                    "nordic walking",
                    "running",
                    "rope jumping",
                ],
                sensor_channels=[
                    "hand_acc_x",
                    "hand_acc_y",
                    "hand_acc_z",
                    "hand_gyro_x",
                    "hand_gyro_y",
                    "hand_gyro_z",
                    "hand_mag_x",
                    "hand_mag_y",
                    "hand_mag_z",
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(0, 7)),
                    val_subj_ids=list(range(7, 8)),
                    test_subj_ids=list(range(8, 9)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                ),
            ),
        ),
    ),
)
