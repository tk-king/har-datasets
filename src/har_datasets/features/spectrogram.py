import os
from typing import List
import numpy as np
import pandas as pd
from scipy.signal import spectrogram  # type: ignore
from tqdm import tqdm

from har_datasets.features.windowing import load_cfg_hash


def get_spectrograms(
    dataset_dir: str,
    cfg_hash: str,
    window_index: pd.DataFrame,
    windows: List[pd.DataFrame],
    sampling_freq: int,
    window_size: int | None,
    overlap: int | None,
    mode: str,
    override_cache: bool,
) -> List[np.ndarray]:
    print("Getting spectrograms...")

    # define dirs
    windowing_dir = os.path.join(dataset_dir, "windowing/")
    spectograms_dir = os.path.join(windowing_dir, "spectograms/")

    # check if windowing corresponds to cfg
    if (
        os.path.exists(windowing_dir)
        and cfg_hash == load_cfg_hash(windowing_dir)
        and not override_cache
    ):
        spectograms = load_spectrograms(windowing_dir, spectograms_dir)

    # if not, generate and save spectrograms
    else:
        spectograms = generate_spectrograms(
            windows, sampling_freq, window_size, overlap, mode
        )

        save_spectrograms(window_index, spectograms_dir, spectograms)

    return spectograms


def save_spectrograms(
    window_index: pd.DataFrame, spectograms_dir: str, spectograms: List[np.ndarray]
) -> None:
    print("Saving spectrograms...")

    # create windowing directory if it does not exist
    os.makedirs(spectograms_dir, exist_ok=True)

    # save spectograms
    loop = tqdm(range(len(window_index)))
    loop.set_description("Saving spectograms")

    for i in loop:
        window_id = window_index.loc[i]["window_id"]
        assert isinstance(window_id, np.integer)

        spect = spectograms[window_id]
        np.save(os.path.join(spectograms_dir, f"spectogram_{i}.npy"), spect)


def load_spectrograms(windowing_dir: str, spectograms_dir: str) -> List[np.ndarray]:
    print("Loading spectrograms...")

    # load window index
    window_index = pd.read_csv(os.path.join(windowing_dir, "window_index.csv"))

    spectograms: List[np.ndarray] = []

    # load spectrograms
    loop = tqdm(range(len(window_index)))
    loop.set_description("Loading spectrograms")

    for i in loop:
        window_id = window_index.loc[i]["window_id"]
        assert isinstance(window_id, np.integer)

        spect = load_spectrogram(spectograms_dir, int(window_id))
        spectograms.append(spect)

    return spectograms


def load_spectrogram(spectograms_dir: str, window_id: int) -> np.ndarray:
    return np.load(os.path.join(spectograms_dir, f"spectogram_{window_id}.npy"))


def generate_spectrograms(
    windows: List[pd.DataFrame],
    sampling_freq: int,
    window_size: int | None,
    overlap: int | None,
    mode: str,
) -> List[np.ndarray]:
    print("Generating spectrograms...")

    spectrograms: List[np.ndarray] = []

    for df in tqdm(windows):
        spect_per_col: List[np.ndarray] = []

        # compute spectrogram for each column
        for col in df.columns:
            # compute spectrogram
            _, _, Sxx = spectrogram(
                x=df[col],
                fs=sampling_freq,
                nperseg=window_size,
                noverlap=overlap,
                mode=mode,
            )

            # store spectrogram
            spect_per_col.append(Sxx)

        # stack for multi-channel spectrogram
        spect = np.stack(spect_per_col, axis=0)

        # store mean spectrogram
        spectrograms.append(spect)

    return spectrograms
