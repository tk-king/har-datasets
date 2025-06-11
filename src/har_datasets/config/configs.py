from har_datasets.config.config import Config, Common, ModelType, SlidingWindow, Dataset

COMMON = Common(
    difference=False,
    datanorm_type=None,
    spectrogram=False,
    model_type=ModelType.freq,
    train_vali_quote=0.8,
    sliding_window=SlidingWindow(sampling_freq=128, windowsize=128, displacement=128),
    wavename="morl",
)


def get_config_uci_har() -> Config:
    dataset: Dataset
    return Config(common=COMMON, dataset=dataset)
