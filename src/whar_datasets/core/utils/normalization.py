from functools import partial
from typing import Callable, Dict, List, Tuple, TypeAlias
import pandas as pd
from whar_datasets.core.config import NormType, WHARConfig
from whar_datasets.core.utils.loading import load_window
from whar_datasets.core.utils.logging import logger

NormParams: TypeAlias = Tuple[Dict[str, float], Dict[str, float]]


def get_normalize(
    cfg: WHARConfig, norm_params: NormParams | None
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    match cfg.normalization:
        case NormType.MIN_MAX_PER_SAMPLE:
            normalize = partial(min_max, norm_params=None)
        case NormType.STD_PER_SAMPLE:
            normalize = partial(standardize, norm_params=None)
        case NormType.ROBUST_SCALE_PER_SAMPLE:
            normalize = partial(robust_scale, norm_params=None)
        case NormType.MIN_MAX_GLOBALLY:
            normalize = partial(min_max, norm_params=norm_params)
        case NormType.STD_GLOBALLY:
            normalize = partial(standardize, norm_params=norm_params)
        case NormType.ROBUST_SCALE_GLOBALLY:
            normalize = partial(robust_scale, norm_params=norm_params)
        case _:
            normalize = partial(load_window)
    return normalize


def get_norm_params(
    cfg: WHARConfig,
    indices: List[int],
    window_metadata: pd.DataFrame,
    windows: Dict[str, pd.DataFrame],
) -> NormParams | None:
    logger.info("Getting normalization parameters...")

    # return None if per sample normalization
    if (
        cfg.normalization == NormType.MIN_MAX_PER_SAMPLE
        or cfg.normalization == NormType.STD_PER_SAMPLE
        or cfg.normalization == NormType.ROBUST_SCALE_PER_SAMPLE
    ):
        return None

    # concat to single df
    windows_df = pd.concat(
        [windows[window_metadata.at[index, "window_id"]] for index in indices],
        ignore_index=True,
    )

    # get normalization params
    match cfg.normalization:
        case NormType.MIN_MAX_GLOBALLY:
            return get_min_max_params(windows_df)
        case NormType.STD_GLOBALLY:
            return get_standardize_params(windows_df)
        case NormType.ROBUST_SCALE_GLOBALLY:
            return get_robust_scale_params(windows_df)
        case _:
            return None


def get_min_max_params(df: pd.DataFrame, exclude_columns: List[str] = []) -> NormParams:
    cols = df.columns.difference(exclude_columns)

    # Compute min and max for each column
    min_values = df[cols].min()
    max_values = df[cols].max()

    # round to 6 decimal places
    min_values = min_values.round(6)
    max_values = max_values.round(6)

    return (min_values.to_dict(), max_values.to_dict())


def get_standardize_params(
    df: pd.DataFrame, exclude_columns: List[str] = []
) -> NormParams:
    cols = df.columns.difference(exclude_columns)

    # Compute mean and standard deviation for each column
    mean_values = df[cols].mean()
    std_values = df[cols].std()

    # round to 6 decimal places
    mean_values = mean_values.round(6)
    std_values = std_values.round(6)

    return (mean_values.to_dict(), std_values.to_dict())


def get_robust_scale_params(
    df: pd.DataFrame, exclude_columns: List[str] = []
) -> NormParams:
    cols = df.columns.difference(exclude_columns)

    # Compute median and IQR (q3 - q1) for each column
    median_values = df[cols].median()
    iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    # round to 6 decimal places
    median_values = median_values.round(6)
    iqr = iqr.round(6)

    return (median_values.to_dict(), iqr.to_dict())


def min_max(
    df: pd.DataFrame, norm_params: NormParams | None, exclude_columns: List[str] = []
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
    df: pd.DataFrame, norm_params: NormParams | None, exclude_columns: List[str] = []
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
    df: pd.DataFrame, norm_params: NormParams | None, exclude_columns: List[str] = []
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
