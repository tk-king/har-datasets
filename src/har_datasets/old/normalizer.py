import numpy as np
import pandas as pd

from har_datasets.schema.schema import NormType


class Normalizer:
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type: NormType):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type

        self.mean: pd.Series
        self.std: pd.Series
        self.min_val: pd.Series
        self.max_val: pd.Series

    def fit(self, df: pd.DataFrame):
        match self.norm_type:
            case NormType.standardization:
                self.mean = df.mean(0)
                self.std = df.std(0)

            case NormType.minmax:
                self.max_val = df.max()
                self.min_val = df.min()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        match self.norm_type:
            case NormType.standardization:
                return (df - self.mean) / (self.std + np.finfo(float).eps)

            case NormType.minmax:
                return (df - self.min_val) / (
                    self.max_val - self.min_val + np.finfo(float).eps
                )

            case NormType.per_sample_std:
                grouped = df.groupby(by=df.index)
                return (df - grouped.transform("mean")) / grouped.transform("std")

            case NormType.per_sample_minmax:
                grouped = df.groupby(by=df.index)
                min_vals = grouped.transform("min")
                return (df - min_vals) / (
                    grouped.transform("max") - min_vals + np.finfo(float).eps
                )
