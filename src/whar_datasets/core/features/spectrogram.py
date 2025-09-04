from typing import List
import numpy as np
import pandas as pd
from scipy.signal import spectrogram  # type: ignore
from tqdm import tqdm
from whar_datasets.core.utils.logging import logger


def generate_spectrograms(
    windows: List[pd.DataFrame],
    sampling_freq: int,
    window_size: int | None,
    overlap: int | None,
    mode: str,
) -> List[np.ndarray]:
    logger.info("Generating spectrograms...")

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
