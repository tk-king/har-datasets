from enum import Enum
from typing import Callable, List, Tuple, TypeAlias, Dict, Optional
from pydantic import BaseModel, field_serializer
import pandas as pd

Parse: TypeAlias = Callable[
    [str, str], Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]
]

NON_CHANNEL_COLS: List[str] = [
    "subject_id",
    "activity_id",
    "session_id",
    "activity_name",
    "timestamp",
]


class NormType(str, Enum):
    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    ROBUST_SCALE_GLOBALLY = "robust_scale_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    ROBUST_SCALE_PER_SAMPLE = "robust_scale_per_sample"


class WHARConfig(BaseModel):
    # Info fields
    dataset_id: str
    download_url: str
    sampling_freq: int
    num_of_subjects: int
    num_of_activities: int
    num_of_channels: int
    datasets_dir: str

    # Parsing fields
    parse: Parse
    activity_id_col: str = "activity_id"

    # Preprocessing fields
    activity_names: List[str]
    sensor_channels: List[str]
    window_time: float  # in seconds
    window_overlap: float  # in [0,1]
    in_parallel: bool = True
    resampling_freq: Optional[int] = None

    # Training fields
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    seed: int = 0
    in_memory: bool = True
    given_train_subj_ids: Optional[List[int]]
    given_test_subj_ids: Optional[List[int]]
    subj_cross_val_split_groups: Optional[List[List[int]]]  # subj_id_groups
    val_percentage: float = 0.1
    normalization: Optional[NormType] = NormType.STD_GLOBALLY

    @field_serializer("parse")
    def serialize_func(self, func, _info):
        return func.__name__
