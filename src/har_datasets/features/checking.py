from typing import List
import pandas as pd
import pandas.api.types as ptypes


def check_format(df: pd.DataFrame, required_cols: List[str]) -> None:
    print("Checking data format...")

    # assert df is not empty
    assert len(df.index) != 0

    # assert required columns are present
    assert set(required_cols).issubset(df.columns)

    # assert required column types
    assert ptypes.is_integer_dtype(df["session_id"])
    assert ptypes.is_integer_dtype(df["activity_id"])
    assert ptypes.is_integer_dtype(df["subject_id"])
    assert ptypes.is_string_dtype(df["activity_name"])
    assert ptypes.is_datetime64_ns_dtype(df["timestamp"])

    # assert channel column types
    channel_cols = [col for col in df.columns if col not in required_cols]
    for c in channel_cols:
        assert ptypes.is_float_dtype(df[c])

    # assert df is sorted by session_id, timestamp
    assert df.sort_values(["session_id", "timestamp"]).equals(df)
