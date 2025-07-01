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


cfg_uci_har = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="real_world",
            download_url="http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",
            sampling_freq=50,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "WALKING",
                    "WALKING_UPSTAIRS",
                    "WALKING_DOWNSTAIRS",
                    "SITTING",
                    "STANDING",
                    "LAYING",
                ],
                sensor_channels=[
                    "total_acc_x",
                    "total_acc_y",
                    "total_acc_z",
                    "body_acc_x",
                    "body_acc_y",
                    "body_acc_z",
                    "body_gyro_x",
                    "body_gyro_y",
                    "body_gyro_z",
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
                    train_subj_ids=list(range(1, 21)),
                    test_subj_ids=list(range(21, 26)),
                    val_subj_ids=list(range(26, 31)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12],
                        [13, 14, 15],
                        [16, 17, 18],
                        [19, 20, 21],
                        [22, 23, 24],
                        [25, 26, 27],
                        [28, 29, 30],
                    ],
                ),
            ),
        ),
    ),
)
