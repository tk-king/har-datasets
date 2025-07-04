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

cfg_daphnet = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="daphnet",
            download_url="https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
            sampling_freq=64,
            num_of_subjects=10,
            num_of_activities=3,
            num_of_channels=9,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "No freeze",
                    "Freeze",
                ],
                sensor_channels=[
                    "shank_acc_x",
                    "shank_acc_y",
                    "shank_acc_z",
                    "thigh_acc_x",
                    "thigh_acc_y",
                    "thigh_acc_z",
                    "trunk_acc_x",
                    "trunk_acc_y",
                    "trunk_acc_z",
                ],
            ),
            sliding_window=SlidingWindow(window_time=1, overlap=0),
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
