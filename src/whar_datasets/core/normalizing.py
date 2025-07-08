from typing import Dict, List, Tuple, TypeAlias
import pandas as pd

from whar_datasets.core.config import NormType, WHARConfig
from whar_datasets.core.sampling import get_window

NormParams: TypeAlias = Tuple[Dict[str, float], Dict[str, float]]


def get_norm_params(
    cfg: WHARConfig,
    indices: List[int],
    windows_dir: str,
    window_metadata: pd.DataFrame,
    windows: Dict[str, pd.DataFrame] | None,
) -> NormParams | None:
    print("Getting normalization parameters...")

    match cfg.dataset.training.normalization:
        case (
            NormType.MIN_MAX_PER_SAMPLE
            | NormType.STD_PER_SAMPLE
            | NormType.ROBUST_SCALE_PER_SAMPLE
        ):
            return None

    # get list of all window dfs
    windows_list = [
        get_window(index, cfg, windows_dir, window_metadata, windows)
        for index in indices
    ]

    # concat to single df
    windows_df = pd.concat(windows_list, ignore_index=True)

    # get normalization params
    match cfg.dataset.training.normalization:
        case NormType.MIN_MAX_GLOBALLY:
            return get_min_max_params(windows_df, ["timestamp"])
        case NormType.STD_GLOBALLY:
            return get_standardize_params(windows_df, ["timestamp"])
        case NormType.ROBUST_SCALE_GLOBALLY:
            return get_robust_scale_params(windows_df, ["timestamp"])
        case _:
            return None


def normalize_window(
    cfg: WHARConfig, norm_params: NormParams | None, window_df: pd.DataFrame
) -> pd.DataFrame:
    match cfg.dataset.training.normalization:
        case NormType.MIN_MAX_PER_SAMPLE:
            return min_max(window_df, ["timestamp"], None)
        case NormType.STD_PER_SAMPLE:
            return standardize(window_df, ["timestamp"], None)
        case NormType.ROBUST_SCALE_PER_SAMPLE:
            return robust_scale(window_df, ["timestamp"], None)
        case NormType.MIN_MAX_GLOBALLY:
            return min_max(window_df, ["timestamp"], norm_params)
        case NormType.STD_GLOBALLY:
            return standardize(window_df, ["timestamp"], norm_params)
        case NormType.ROBUST_SCALE_GLOBALLY:
            return robust_scale(window_df, ["timestamp"], norm_params)
        case _:
            return window_df


def min_max(
    df: pd.DataFrame,
    exclude_columns: List[str],
    norm_params: NormParams | None,
) -> pd.DataFrame:
    norm_params = (
        get_min_max_params(df, exclude_columns) if norm_params is None else norm_params
    )

    min_values = pd.Series(norm_params[0])
    max_values = pd.Series(norm_params[1])

    # Apply min-max normalization
    df_normalized = (df - min_values) / (max_values - min_values)

    return df_normalized


def standardize(
    df: pd.DataFrame,
    exclude_columns: List[str],
    norm_params: NormParams | None,
) -> pd.DataFrame:
    norm_params = (
        get_standardize_params(df, exclude_columns)
        if norm_params is None
        else norm_params
    )

    mean_values = pd.Series(norm_params[0])
    std_values = pd.Series(norm_params[1])

    # Apply standardization
    df_normalized = (df - mean_values) / std_values

    return df_normalized


def robust_scale(
    df: pd.DataFrame,
    exclude_columns: List[str],
    norm_params: NormParams | None,
) -> pd.DataFrame:
    norm_params = (
        get_robust_scale_params(df, exclude_columns)
        if norm_params is None
        else norm_params
    )

    median_values = pd.Series(norm_params[0])
    iqr = pd.Series(norm_params[1])

    # Apply robust scaling
    df_normalized = (df - median_values) / iqr

    return df_normalized


def get_min_max_params(df: pd.DataFrame, exclude_columns: List[str]) -> NormParams:
    cols = df.columns.difference(exclude_columns)

    # Compute min and max for each column
    min_values = df[cols].min()
    max_values = df[cols].max()

    return (min_values.to_dict(), max_values.to_dict())


def get_standardize_params(df: pd.DataFrame, exclude_columns: List[str]) -> NormParams:
    cols = df.columns.difference(exclude_columns)

    # Compute mean and standard deviation for each column
    mean_values = df[cols].mean()
    std_values = df[cols].std()

    return (mean_values.to_dict(), std_values.to_dict())


def get_robust_scale_params(df: pd.DataFrame, exclude_columns: List[str]) -> NormParams:
    cols = df.columns.difference(exclude_columns)

    # Compute median and IQR (q3 - q1) for each column
    median_values = df[cols].median()
    iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    return (median_values.to_dict(), iqr.to_dict())
