from enum import Enum
from typing import Callable, List, Tuple, TypeAlias, Dict, Optional
from pydantic import BaseModel, field_serializer
import pandas as pd

Parse: TypeAlias = Callable[
    [str, str], Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]
]


class NormType(str, Enum):
    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    ROBUST_SCALE_GLOBALLY = "robust_scale_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    ROBUST_SCALE_PER_SAMPLE = "robust_scale_per_sample"


class TransformType(Enum):
    DWT = "dwt"
    STFT = "stft"


class WHARConfig(BaseModel):
    # metadata fields
    dataset_id: str
    download_url: str
    sampling_freq: int
    num_of_subjects: int
    num_of_activities: int
    num_of_channels: int

    # flow fields
    datasets_dir: str
    in_memory: bool = True
    parallelize: bool = False

    # parsing fields
    parse: Parse
    activity_id_col: str = "activity_id"

    # preprocessing fields
    activity_names: List[str]
    sensor_channels: List[str]
    window_time: float  # in seconds
    window_overlap: float  # in [0,1]
    resampling_freq: Optional[int] = None

    # postprocessing fields
    given_train_test_subj_ids: Optional[Tuple[List[int], List[int]]]
    subj_cross_val_split_groups: Optional[List[List[int]]]
    val_percentage: float = 0.1
    normalization: Optional[NormType] = NormType.STD_GLOBALLY
    transform: Optional[TransformType] = None

    # training fields
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    seed: int = 0

    @field_serializer("parse")
    def serialize_func(self, func, _info):
        return func.__name__
