from enum import Enum
from typing import List
from pydantic import BaseModel

NON_CHANNEL_COLS: List[str] = [
    "subject_id",
    "activity_id",
    "session_id",
    "activity_name",
    "timestamp",
]


class NormType(Enum):
    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    STD_PER_SUBJ = "std_per_subj"
    MIN_MAX_PER_SUBJ = "min_max_per_subj"


class GivenSplit(BaseModel):
    train_subj_ids: List[int]  # list of subject ids in train set
    test_subj_ids: List[int]  # list of subject ids in test set
    val_subj_ids: List[int]  # list of subject ids in val set


class SubjCrossValSplit(BaseModel):
    subj_id_groups: List[List[int]]  # groups containing multiple subject ids


class Split(BaseModel):
    given_split: GivenSplit | None  # how to split subjects into train / test / val
    subj_cross_val_split: SubjCrossValSplit | None  # split based on groups


class Training(BaseModel):
    batch_size: int  # batch size of train loader
    learning_rate: float
    num_epochs: int
    seed: int = 0
    shuffle: bool = True  # whether to shuffle train loader
    in_memory: bool = True
    split: Split  # how to split into train / test / val


class Selections(BaseModel):
    activity_names: List[str]  # list of activity names to include
    sensor_channels: List[str]  # list of channels to include


class SlidingWindow(BaseModel):
    window_time: float  # in seconds
    overlap: float  # in [0, 1]


class Caching(BaseModel):
    cache_parsed: bool = True
    cache_windows: bool = True
    cache_spectrograms: bool = True


class Preprocessing(BaseModel):
    activity_id_col: str = "activity_id"
    selections: Selections  # which activities and channels to include
    normalization: NormType | None = None  # type of normalization to apply to all
    sliding_window: SlidingWindow
    caching: Caching = Caching()


class Info(BaseModel):
    id: str  # id of the dataset
    download_url: str  # url to download dataset
    sampling_freq: int  # sampling frequency of the dataset


class Dataset(BaseModel):
    info: Info
    preprocessing: Preprocessing
    training: Training


class Spectrogram(BaseModel):
    window_size: int | None = 32
    overlap: int | None = None
    mode: str = "magnitude"


class Common(BaseModel):
    datasets_dir: str  # directory to save all datasets
    resampling_freq: int | None = None  # common sampling frequency to which to convert
    use_derivative: bool = False
    use_spectrogram: bool = False
    spectrogram: Spectrogram = Spectrogram()


class HARConfig(BaseModel):
    common: Common  # common config applyed to all datasets
    dataset: Dataset  # dataset specific config
