from enum import Enum
from typing import List
from pydantic import BaseModel


class NormType(Enum):
    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    STD_PER_SUBJ = "std_per_subj"
    MIN_MAX_PER_SUBJ = "min_max_per_subj"


class FeaturesType(Enum):
    CHANNELS_ONLY = "channels_only"
    FREQS_ONLY = "freqs_only"
    BOTH = "both"


class SplitType(Enum):
    GIVEN = "given"
    SUBJ_CROSS_VAL = "subj_cross_val"


class GivenSplit(BaseModel):
    train_subj_ids: List[int]  # list of subject ids in train set
    test_subj_ids: List[int]  # list of subject ids in test set
    val_subj_ids: List[int]  # list of subject ids in val set


class SubjCrossValSplit(BaseModel):
    subj_id_group_index: int  # current group selected for testing
    subj_id_groups: List[List[int]]  # groups containing multiple subject ids


class Split(BaseModel):
    split_type: SplitType  # type of split, either given or subj cross val
    given_split: GivenSplit | None  # how to split subjects into train / test / val
    subj_cross_val_split: SubjCrossValSplit | None  # split based on groups


class Training(BaseModel):
    batch_size: int  # batch size of train loader
    shuffle: bool  # whether to shuffle train loader
    learning_rate: float
    num_epochs: int


class Selections(BaseModel):
    activity_names: List[str]  # list of activity names to include
    channels: List[str]  # list of channels to include


class Info(BaseModel):
    url: str  # url to download dataset
    csv_file: str  # name of the csv file in central format
    sampling_freq: int  # sampling frequency of the dataset


class Dataset(BaseModel):
    info: Info  # info about the dataset
    selections: Selections  # which activities and channels to include
    split: Split  # how to split into train / test / val
    training: Training  # training config


class SlidingWindow(BaseModel):
    window_time: float  # in seconds
    overlap: float  # in [0, 1]


class Common(BaseModel):
    datasets_dir: str  # directory to save all datasets
    resampling_freq: int | None  # common sampling frequency to which to convert
    normalization: NormType | None  # type of normalization to apply to all
    features_type: FeaturesType  # type of features to use
    include_derivative: bool  # whether to include derivative features
    sliding_window: SlidingWindow  # common sliding window config


class HARConfig(BaseModel):
    common: Common  # common config applyed to all datasets
    dataset: Dataset  # dataset specific config
