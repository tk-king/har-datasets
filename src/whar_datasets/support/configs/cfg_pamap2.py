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
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "lying",
                    "sitting",
                    "standing",
                    "walking",
                    "running",
                    "cycling",
                    "nordic walking",
                    "watching TV",
                    "computer work",
                    "car driving",
                    "ascending stairs",
                    "descending stairs",
                    "vacuum cleaning",
                    "ironing",
                    "folding laundry",
                    "house cleaning",
                    "playing soccer",
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
            batch_size=32,
            learning_rate=0.0001,
            num_epochs=100,
            split=Split(
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
