from typing import List
import pandas as pd

EXCLUDE_COLUMNS = ["subj_id", "activity_id", "activity_block_id", "activity_name"]


def normalize_min_max(
    df: pd.DataFrame, exclude_columns: List[str] = EXCLUDE_COLUMNS
) -> pd.DataFrame:
    # Compute min and max for each column
    min_values = df[df.columns.difference(exclude_columns)].min()
    max_values = df[df.columns.difference(exclude_columns)].max()

    # Apply min-max normalization
    df_normalized = (df - min_values) / (max_values - min_values)

    return df_normalized


def standardize(
    df: pd.DataFrame, exclude_columns: List[str] = EXCLUDE_COLUMNS
) -> pd.DataFrame:
    # Compute mean and standard deviation for each column
    mean_values = df[df.columns.difference(exclude_columns)].mean()
    std_values = df[df.columns.difference(exclude_columns)].std()

    # Apply standardization
    df_normalized = (df - mean_values) / std_values

    return df_normalized


def robust_scale(
    df: pd.DataFrame,
    exclude_columns: List[str] = EXCLUDE_COLUMNS,
) -> pd.DataFrame:
    # Compute median and IQR (q3 - q1) for each column
    median_values = df[df.columns.difference(exclude_columns)].median()
    q1 = df[df.columns.difference(exclude_columns)].quantile(0.25)
    q3 = df[df.columns.difference(exclude_columns)].quantile(0.75)

    # Apply robust scaling
    df_normalized = (df - median_values) / (q3 - q1)

    return df_normalized
