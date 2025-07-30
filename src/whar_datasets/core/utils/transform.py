from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from whar_datasets.core.config import WHARConfig, TransformType
from whar_datasets.core.features.wavelet_transform import signal_to_dwt_grid


def transform_windows_seq(
    cfg: WHARConfig, window_metadata: pd.DataFrame, normalized: Dict[str, np.ndarray]
) -> Dict[str, List[np.ndarray]]:
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Transforming windows")

    transform: Callable[[Any], List[np.ndarray]]

    match cfg.transform:
        case TransformType.DWT:

            def transform_dwt(x):
                grid, lengths = signal_to_dwt_grid(x)
                return [grid, np.array(lengths)]

            transform = transform_dwt
        case _:
            return {window_id: [] for window_id in loop}

    return {window_id: transform(normalized[window_id]) for window_id in loop}
