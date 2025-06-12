from typing import List

import pandas as pd


def resample(
    df: pd.DataFrame,
    sampling_freq: float,
    resampling_freq: float,
    exclude_columns: List[str],
) -> pd.DataFrame:
    resampled_sessions = []

    for session_id, group in df.groupby("session_id"):
        group = group.reset_index(drop=True)

        # Generate synthetic timestamp index in milliseconds
        dt_ms = int(1000 / sampling_freq)
        group["timestamp"] = pd.to_datetime(group.index * dt_ms, unit="ms")
        group = group.set_index("timestamp")

        # Columns to resample (sensor data)
        resample_cols = [col for col in group.columns if col not in exclude_columns]

        # Resample to new frequency
        new_period_ms = int(1000 / resampling_freq)
        resampled = (
            group[resample_cols].resample(f"{new_period_ms}ms").mean().interpolate()
        )

        # Reattach excluded columns via forward fill
        for col in exclude_columns:
            if col in group.columns:
                resampled[col] = group[col].resample(f"{new_period_ms}ms").ffill()

        # Reset index and add session_id back
        resampled = resampled.reset_index(drop=True)
        resampled["session_id"] = session_id

        resampled_sessions.append(resampled)

    return pd.concat(resampled_sessions, ignore_index=True)
