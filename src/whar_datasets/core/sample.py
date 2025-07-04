from typing import List
import numpy as np
import pandas as pd
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.features.spectrogram import load_spectrogram
from whar_datasets.core.utils.loading import load_window


def get_label(
    index: int,
    window_index: pd.DataFrame,
    session_index: pd.DataFrame,
) -> np.integer:
    # get session_id
    session_id = window_index.iloc[index]["session_id"]
    assert isinstance(session_id, np.integer)

    # get activity id from sessionindex
    label = session_index.loc[
        session_index["session_id"] == session_id, "activity_id"
    ].values[0]
    assert isinstance(label, np.integer)

    return label


def get_window(
    index: int,
    cfg: WHARConfig,
    windows_dir: str,
    window_index: pd.DataFrame,
    windows: List[pd.DataFrame] | None,
) -> np.ndarray:
    # get window_id
    window_id = window_index.iloc[index]["window_id"]
    assert isinstance(window_id, str)

    # select or load window
    window = (
        windows[index].values
        if windows is not None and cfg.dataset.training.in_memory
        else load_window(windows_dir, window_id).values
    )

    return window


def get_spectrogram(
    index: int,
    cfg: WHARConfig,
    spectograms_dir: str,
    window_index: pd.DataFrame,
    spectograms: List[np.ndarray] | None,
) -> np.ndarray:
    # get window_id
    window_id = window_index.iloc[index]["window_id"]
    assert isinstance(window_id, np.integer)

    # select or load spectogram
    spect = (
        spectograms[window_id]
        if spectograms is not None and cfg.dataset.training.in_memory
        else load_spectrogram(spectograms_dir, int(window_id))
    )

    return spect
