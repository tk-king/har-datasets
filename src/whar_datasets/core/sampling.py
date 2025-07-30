from typing import Dict, List
import numpy as np
import pandas as pd
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.loading import load_sample, load_window


def get_label(
    index: int,
    window_metadata: pd.DataFrame,
    session_metadata: pd.DataFrame,
) -> int:
    # get session_id
    session_id = int(window_metadata.at[index, "session_id"])
    assert isinstance(session_id, int)

    # get activity_id from session_metadata
    label = session_metadata.loc[
        session_metadata["session_id"] == session_id, "activity_id"
    ].item()
    assert isinstance(label, int)

    return label


def get_window(
    index: int,
    cfg: WHARConfig,
    windows_dir: str,
    window_metadata: pd.DataFrame,
    windows: Dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    # get window_id
    window_id = window_metadata.at[index, "window_id"]
    assert isinstance(window_id, str)

    # select or load window
    window = (
        windows[window_id]
        if windows is not None and cfg.in_memory
        else load_window(windows_dir, window_id)
    )

    return window


def get_sample(
    index: int,
    cfg: WHARConfig,
    samples_dir: str,
    window_metadata: pd.DataFrame,
    samples: Dict[str, List[np.ndarray]] | None,
) -> List[np.ndarray]:
    # get window_id
    window_id = window_metadata.at[index, "window_id"]
    assert isinstance(window_id, str)

    # select or load sample
    sample = (
        samples[window_id]
        if samples is not None and cfg.in_memory
        else load_sample(samples_dir, window_id)
    )

    return sample
