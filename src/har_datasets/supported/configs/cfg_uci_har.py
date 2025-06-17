from har_datasets.config.config import (
    Common,
    Dataset,
    FeaturesType,
    GivenSplit,
    HARConfig,
    Info,
    Selections,
    SlidingWindow,
    Spectrogram,
    Split,
    SplitType,
    SubjCrossValSplit,
    Training,
)


cfg_uci_har = HARConfig(
    common=Common(
        datasets_dir="./datasets",
        resampling_freq=None,
        normalization=None,
        features_type=FeaturesType.CHANNELS_ONLY,
        include_derivative=False,
        sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        spectrogram=Spectrogram(window_size=20, overlap=None, mode="magnitude"),
    ),
    dataset=Dataset(
        info=Info(
            id="uci_har",
            url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
            sampling_freq=50,
        ),
        selections=Selections(
            activity_names=[
                "WALKING",
                "WALKING_UPSTAIRS",
                "WALKING_DOWNSTAIRS",
                "SITTING",
                "STANDING",
                "LAYING",
            ],
            channels=[
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
        split=Split(
            split_type=SplitType.GIVEN,
            given_split=GivenSplit(
                train_subj_ids=list(range(1, 21)),
                test_subj_ids=[21, 22, 23, 24, 25],
                val_subj_ids=[26, 27, 28, 29, 30],
            ),
            subj_cross_val_split=SubjCrossValSplit(
                subj_id_group_index=0,
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
        training=Training(
            batch_size=32,
            shuffle=True,
            learning_rate=0.0001,
            num_epochs=100,
        ),
    ),
)
