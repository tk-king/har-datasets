from typing import Callable, List
import pandas as pd


def normalize_globally(
    df: pd.DataFrame,
    normalize: Callable[[pd.DataFrame, List[str]], pd.DataFrame],
    exclude_columns: List[str],
) -> pd.DataFrame:
    return normalize(df, exclude_columns)


def normalize_per_subject(
    df: pd.DataFrame,
    normalize: Callable[[pd.DataFrame, List[str]], pd.DataFrame],
    exclude_columns: List[str],
) -> pd.DataFrame:
    return df.groupby("subject_id", group_keys=False).transform(
        lambda x: normalize(x, exclude_columns)
    )


def normalize_per_sample(
    windows: List[pd.DataFrame],
    normalize: Callable[[pd.DataFrame, List[str]], pd.DataFrame],
    exclude_columns: List[str],
) -> List[pd.DataFrame]:
    return [normalize(window, exclude_columns) for window in windows]


def min_max(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    cols = df.columns.difference(exclude_columns)

    # Compute min and max for each column
    min_values = df[cols].min()
    max_values = df[cols].max()

    # Apply min-max normalization
    df_normalized = (df - min_values) / (max_values - min_values)

    return df_normalized


def standardize(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    cols = df.columns.difference(exclude_columns)

    # Compute mean and standard deviation for each column
    mean_values = df[cols].mean()
    std_values = df[cols].std()

    # Apply standardization
    df_normalized = (df - mean_values) / std_values

    return df_normalized


def robust_scale(df: pd.DataFrame, exclude_columns: List[str]) -> pd.DataFrame:
    cols = df.columns.difference(exclude_columns)

    # Compute median and IQR (q3 - q1) for each column
    median_values = df[cols].median()
    iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    # Apply robust scaling
    df_normalized = (df - median_values) / iqr

    return df_normalized
