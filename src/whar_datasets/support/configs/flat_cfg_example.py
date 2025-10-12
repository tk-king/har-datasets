from whar_datasets.core.config import NormType, WHARConfig


my_dataset_cfg = WHARConfig(
    # metadata
    dataset_id="my_dataset",
    download_url="https://example.com/my_dataset.zip",
    datasets_dir="./datasets/",
    sampling_freq=50,
    num_of_subjects=6,
    num_of_activities=3,
    num_of_channels=3,
    # preprocessing
    parse=parse_my_dataset,
    activity_names=["walking", "jumping", "sitting"],
    sensor_channels=["accel_x", "accel_y", "accel_z"],
    window_time=2.56,  # in seconds
    window_overlap=0.5,  # in [0,1]
    resampling_freq=None,
    # postprocessing
    given_fold=([0, 1, 2, 3], [4, 5]),
    fold_groups=[[0, 1], [2, 3], [4, 5]],
    val_percentage=0.1,  # in [0,1]
    normalization=NormType.STD_GLOBALLY,
    transform=None,
    # efficiency
    in_memory=True,
    parallelize=True,
    # training
    batch_size=64,
    learning_rate=1e-4,
    num_epochs=100,
    seed=0,
)
