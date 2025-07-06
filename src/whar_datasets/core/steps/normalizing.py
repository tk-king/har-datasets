from typing import Callable, Dict, List
import pandas as pd


def normalize_per_session(
    session_df: pd.DataFrame,
    normalize: Callable[[pd.DataFrame, List[str]], pd.DataFrame] | None,
) -> pd.DataFrame:
    # print("Normalizing data per session...")

    if normalize is None:
        return session_df
    else:
        return normalize(session_df, ["timestamp"])


def normalize_per_sample(
    window_dfs: Dict[str, pd.DataFrame],
    normalize: Callable[[pd.DataFrame, List[str]], pd.DataFrame] | None,
) -> Dict[str, pd.DataFrame]:
    # print("Normalizing data per sample...")

    if normalize is None:
        return window_dfs
    else:
        return {
            window_id: normalize(window_df, ["timestamp"])
            for window_id, window_df in window_dfs.items()
        }


def min_max(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    # print("Normalizing with min-max normalization...")

    cols = df.columns.difference(exclude_columns)

    # Compute min and max for each column
    min_values = df[cols].min()
    max_values = df[cols].max()

    # Apply min-max normalization
    df_normalized = (df - min_values) / (max_values - min_values)

    return df_normalized


def standardize(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    # print("Normalizing with standardization...")

    cols = df.columns.difference(exclude_columns)

    # Compute mean and standard deviation for each column
    mean_values = df[cols].mean()
    std_values = df[cols].std()

    # Apply standardization
    df_normalized = (df - mean_values) / std_values

    return df_normalized


def robust_scale(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    # print("Normalizing with robust scaling...")

    cols = df.columns.difference(exclude_columns)

    # Compute median and IQR (q3 - q1) for each column
    median_values = df[cols].median()
    iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    # Apply robust scaling
    df_normalized = (df - median_values) / iqr

    return df_normalized
