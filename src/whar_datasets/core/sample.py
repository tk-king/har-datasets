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
    # get window_id
    window_id = window_index.loc[index]["window_id"]
    assert isinstance(window_id, np.integer)

    # get activtiy id from sessionindex where window_id fits
    label = session_index[session_index["window_id"] == window_id]["activity_id"]
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
    window_id = window_index.loc[index]["window_id"]
    assert isinstance(window_id, np.integer)

    # select or load window
    window = (
        windows[window_id].values
        if windows is not None and cfg.dataset.training.in_memory
        else load_window(windows_dir, int(window_id)).values
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
    window_id = window_index.loc[index]["window_id"]
    assert isinstance(window_id, np.integer)

    # select or load spectogram
    spect = (
        spectograms[window_id]
        if spectograms is not None and cfg.dataset.training.in_memory
        else load_spectrogram(spectograms_dir, int(window_id))
    )

    return spect
