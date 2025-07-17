from typing import Dict, List, Tuple, TypeAlias
import pandas as pd
from tqdm import tqdm
# from dask.delayed import delayed
# from dask.base import compute
# from dask.diagnostics.progress import ProgressBar

from whar_datasets.core.config import NormType, WHARConfig
from whar_datasets.core.sampling import get_window

# from whar_datasets.core.utils.caching import cache_norm_params_hash, cache_windows
# from whar_datasets.core.utils.hashing import (
#     create_norm_params_hash,
#     load_norm_params_hash,
# )
from whar_datasets.core.utils.loading import load_window, load_windows

NormParams: TypeAlias = Tuple[Dict[str, float], Dict[str, float]]


def normalize_windows(
    cfg: WHARConfig,
    indices: List[int],
    hashes_dir: str,
    windows_dir: str,
    window_metadata: pd.DataFrame,
    override_cache: bool,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame] | None]:
    windows = load_windows(window_metadata, windows_dir)
    norm_params = get_norm_params(cfg, indices, windows_dir, window_metadata, windows)

    # # check if already cached
    # norm_params_hash = create_norm_params_hash(norm_params)
    # if norm_params_hash == load_norm_params_hash(hashes_dir) and not override_cache:
    #     return window_metadata, windows

    # normalize windows
    pairs = normalize_windows_sequentially(
        cfg, norm_params, windows_dir, window_metadata, windows
    )

    # construct normalized_windows
    norm_windows = {window_id: window_df for window_id, window_df in pairs}

    # get new normalization parameters
    # new_norm_params = get_norm_params(
    #     cfg, indices, windows_dir, window_metadata, norm_windows
    # )
    # new_norm_params_hash = create_norm_params_hash(new_norm_params)

    # cache normalized windows
    # cache_windows(windows_dir, window_metadata, norm_windows)
    # cache_norm_params_hash(hashes_dir, new_norm_params_hash)

    if cfg.dataset.training.in_memory:
        return window_metadata, norm_windows

    return window_metadata, None


def normalize_windows_sequentially(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    windows_dir: str,
    window_metadata: pd.DataFrame,
    windows: Dict[str, pd.DataFrame] | None,
) -> List[Tuple[str, pd.DataFrame]]:
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Normalizing windows")

    pairs = [
        normalize_window(cfg, norm_params, windows, windows_dir, window_id)
        for window_id in loop
    ]

    return pairs


# def normalize_windows_parallely(
#     cfg: WHARConfig,
#     norm_params: NormParams | None,
#     windows_dir: str,
#     window_metadata: pd.DataFrame,
# ) -> List[Tuple[str, pd.DataFrame]]:
#     @delayed
#     def normalize_window_delayed(window_id: str) -> Tuple[str, pd.DataFrame]:
#         assert isinstance(window_id, str)

#         window_df = load_window(windows_dir, window_id)

#         match cfg.dataset.training.normalization:
#             case NormType.MIN_MAX_PER_SAMPLE:
#                 return window_id, min_max(window_df, None)
#             case NormType.STD_PER_SAMPLE:
#                 return window_id, standardize(window_df, None)
#             case NormType.ROBUST_SCALE_PER_SAMPLE:
#                 return window_id, robust_scale(window_df, None)
#             case NormType.MIN_MAX_GLOBALLY:
#                 return window_id, min_max(window_df, norm_params)
#             case NormType.STD_GLOBALLY:
#                 return window_id, standardize(window_df, norm_params)
#             case NormType.ROBUST_SCALE_GLOBALLY:
#                 return window_id, robust_scale(window_df, norm_params)
#             case _:
#                 return window_id, window_df

#     # define processing tasks
#     tasks = (
#         normalize_window_delayed(window_id)
#         for window_id in window_metadata["window_id"]
#     )

#     # execute tasks in parallel
#     pbar = ProgressBar()
#     pbar.register()
#     pairs = list(compute(*tasks, scheduler="processes"))
#     pbar.unregister()

#     return pairs


def normalize_window(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    windows: Dict[str, pd.DataFrame] | None,
    windows_dir: str,
    window_id: str,
) -> Tuple[str, pd.DataFrame]:
    assert isinstance(window_id, str)

    window_df = (
        windows[window_id]
        if windows is not None
        else load_window(windows_dir, window_id)
    )

    match cfg.dataset.training.normalization:
        case NormType.MIN_MAX_PER_SAMPLE:
            return window_id, min_max(window_df, None)
        case NormType.STD_PER_SAMPLE:
            return window_id, standardize(window_df, None)
        case NormType.ROBUST_SCALE_PER_SAMPLE:
            return window_id, robust_scale(window_df, None)
        case NormType.MIN_MAX_GLOBALLY:
            return window_id, min_max(window_df, norm_params)
        case NormType.STD_GLOBALLY:
            return window_id, standardize(window_df, norm_params)
        case NormType.ROBUST_SCALE_GLOBALLY:
            return window_id, robust_scale(window_df, norm_params)
        case _:
            return window_id, window_df


def get_norm_params(
    cfg: WHARConfig,
    indices: List[int],
    windows_dir: str,
    window_metadata: pd.DataFrame,
    windows: Dict[str, pd.DataFrame] | None,
) -> NormParams | None:
    print("Getting normalization parameters...")

    # return None if per sample normalization
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
    df: pd.DataFrame,
    norm_params: NormParams | None,
    exclude_columns: List[str] = [],
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
    norm_params: NormParams | None,
    exclude_columns: List[str] = [],
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
    norm_params: NormParams | None,
    exclude_columns: List[str] = [],
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
