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

cfg_ku_har = HARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="ku_har",
            download_url="https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded",
            sampling_freq=100,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "Stand",
                    "Sit",
                    "Talk-sit",
                    "Talk-stand",
                    "Stand-sit",
                    "Lay",
                    "Lay-stand",
                    "Pick",
                    "Jump",
                    "Push-up",
                    "Sit-up",
                    "Walk",
                    "Walk-backward",
                    "Walk-circle",
                    "Run",
                    "Stair-up",
                    "Stair-down",
                    "Table-tennis",
                ],
                sensor_channels=[
                    "acc_x",
                    "acc_y",
                    "acc_z",
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
                given_split=GivenSplit(
                    train_subj_ids=list(range(0, 60)),
                    val_subj_ids=list(range(60, 75)),
                    test_subj_ids=list(range(75, 90)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[
                        list(range(start, start + 10)) for start in range(0, 90)
                    ],
                ),
            ),
        ),
    ),
)
