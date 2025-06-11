from enum import Enum
from typing import Dict, List, Tuple
from pydantic import BaseModel


class NormType(Enum):
    standardization = "standardization"
    minmax = "minmax"
    per_sample_std = "per_sample_std"
    per_sample_minmax = "per_sample_minmax"


class ModelType(Enum):
    freq = "freq"
    cross = "cross"


class ExpMode(Enum):
    Given = "Given"
    LSOCV = "LOCV"


class Keys(BaseModel):
    all: List[int]
    train: List[int]
    val: List[int]
    test: List[int]
    LOCV: List[Tuple[int, ...]]


class Training(BaseModel):
    batch_size: int
    shuffle: bool
    drop_last: bool
    learning_rate: float
    epochs: int


class Dataset(BaseModel):
    file_name: str
    dir: str
    freq_save_path: str
    used_cols: Dict[int, str]
    label_map: Dict[int, Tuple[str, bool]]
    keys: Keys
    locv: Dict[str, str]
    exp_mode: ExpMode
    split_tag: str
    file_encoding: Dict[int, str]
    training: Training


class SlidingWindow(BaseModel):
    sampling_freq: int
    windowsize: int
    displacement: int


class Common(BaseModel):
    difference: bool
    datanorm_type: NormType | None
    spectrogram: bool
    model_type: ModelType
    train_vali_quote: float
    sliding_window: SlidingWindow
    wavename: str


class Config(BaseModel):
    common: Common
    dataset: Dataset
