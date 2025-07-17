from whar_datasets.core.config import NormType, WHARConfig


cfg = WHARConfig(
    # Info
    dataset_id="example",
    download_url="https://example.zip",
    sampling_freq=50,
    num_of_subjects=30,
    num_of_activities=6,
    num_of_channels=9,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_example,
    # Preprocessing
    activity_names=["walking", "sitting", "laying"],
    sensor_channels=["acc_x", "acc_y", "acc_z"],
    window_time=2.56,
    window_overlap=0.5,
    in_parallel=True,
    resampling_freq=None,
    # Training
    batch_size=64,
    learning_rate=1e-4,
    num_epochs=100,
    seed=0,
    in_memory=True,
    given_train_subj_ids=list(range(0, 24)),
    given_test_subj_ids=list(range(24, 30)),
    subj_cross_val_split_groups=[
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ],
    val_percentage=0.1,
    normalization=NormType.STD_GLOBALLY,
)
