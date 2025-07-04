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

cfg_har_sense = WHARConfig(
    common=Common(
        datasets_dir="./datasets",
    ),
    dataset=Dataset(
        info=Info(
            id="har_sense",
            download_url="https://www.kaggle.com/api/v1/datasets/download/nurulaminchoudhury/harsense-datatset",
            sampling_freq=50,
            num_of_subjects=12,
            num_of_activities=6,
            num_of_channels=16,
        ),
        preprocessing=Preprocessing(
            selections=Selections(
                activity_names=[
                    "Walking",
                    "Standing",
                    "Upstairs",
                    "Downstairs",
                    "Running",
                    "Sitting",
                ],
                sensor_channels=[
                    "AG-X",
                    "AG-Y",
                    "AG-Z",
                    "Acc-X",
                    "Acc-Y",
                    "Acc-Z",
                    "Gravity-X",
                    "Gravity-Y",
                    "Gravity-Z",
                    "RR-X",
                    "RR-Y",
                    "RR-Z",
                    "RV-X",
                    "RV-Y",
                    "RV-Z",
                    "cos",
                ],
            ),
            sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        ),
        training=Training(
            split=Split(
                given_split=GivenSplit(
                    train_subj_ids=list(range(1, 9)),
                    val_subj_ids=list(range(9, 11)),
                    test_subj_ids=list(range(11, 13)),
                ),
                subj_cross_val_split=SubjCrossValSplit(
                    subj_id_groups=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                ),
            ),
        ),
    ),
)
