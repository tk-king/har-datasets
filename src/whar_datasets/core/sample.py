import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.features.spectrogram import load_spectrogram
from whar_datasets.core.steps.windowing import load_window


def get_sample(
    cfg: WHARConfig,
    index: int,
    dataset_dir: str,
    window_index: pd.DataFrame,
    windows: List[pd.DataFrame] | None,
    spectograms: List[np.ndarray] | None,
) -> Tuple[np.integer, np.ndarray, np.ndarray | None]:
    # get class label of window
    label = window_index.loc[index]["activity_id"]
    assert isinstance(label, np.integer)

    # get window_id
    window_id = window_index.loc[index]["window_id"]
    assert isinstance(window_id, np.integer)

    if cfg.dataset.training.in_memory:
        assert windows is not None

        # get window and spectogram
        window = windows[window_id].values
        spect = (
            spectograms[window_id]
            if cfg.common.use_spectrogram and spectograms is not None
            else None
        )
    else:
        # defined dirs
        windowing_dir = os.path.join(dataset_dir, "windowing/")
        windows_dir = os.path.join(windowing_dir, "windows/")
        spectograms_dir = os.path.join(windowing_dir, "spectograms/")

        # load window and spectogram
        window = load_window(windows_dir, int(window_id)).values
        spect = (
            load_spectrogram(spectograms_dir, int(window_id))
            if cfg.common.use_spectrogram
            else None
        )

    return label, window, spect
