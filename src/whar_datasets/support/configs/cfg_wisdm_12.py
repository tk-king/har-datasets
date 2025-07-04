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

cfg_wisdm_12 = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="wisdm_12",
            download_url="https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
            sampling_freq=20,
            num_of_subjects=36,
            num_of_activities=6,
            num_of_channels=3,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "Walking",
                    "Jogging",
                    "Upstairs",
                    "Downstairs",
                    "Sitting",
                    "Standing",
                ],
                sensor_channels=[
                    "accel_x",
                    "accel_y",
                    "accel_z",
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(0, 26)),
                    val_subj_ids=list(range(26, 31)),
                    test_subj_ids=list(range(31, 36)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[
                        [0, 1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16, 17],
                        [18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34, 35],
                    ],
                ),
            ),
        ),
    ),
)
