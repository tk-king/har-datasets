from enum import Enum
from typing import List
from pydantic import BaseModel


class NormType(Enum):
    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    STD_PER_SUBJECT = "std_per_subject"
    MIN_MAX_PER_SUBJECT = "min_max_per_subject"


class FeaturesType(Enum):
    CHANNELS_ONLY = "channels_only"
    FREQUENCIES_ONLY = "frequencies_only"
    BOTH = "both"


class SplitType(Enum):
    GIVEN = "given"
    SCV = "SCV"


class GivenSplit(BaseModel):
    train_subject_ids: List[int]
    test_subject_ids: List[int]
    val_subject_ids: List[int]


class SCVSplit(BaseModel):
    test_group_index: int
    test_groups: List[List[int]]


class Split(BaseModel):
    split_type: SplitType
    given_split: GivenSplit
    scv_split: SCVSplit


class Training(BaseModel):
    batch_size: int
    shuffle: bool
    learning_rate: float
    num_epochs: int


class Selections(BaseModel):
    activity_ids: List[int]
    channels: List[str]


class Dataset(BaseModel):
    url: str
    dir: str
    csv_file: str
    sampling_freq: int
    selections: Selections
    split: Split
    training: Training


class SlidingWindow(BaseModel):
    window_size: int
    displacement: int


class Common(BaseModel):
    resampling_freq: int | None
    normalization: NormType | None
    features_type: FeaturesType
    include_derivative: bool
    sliding_window: SlidingWindow


class Config(BaseModel):
    common: Common
    dataset: Dataset
