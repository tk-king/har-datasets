from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.normalization import NormParams, get_normalize
from whar_datasets.processing.utils.transform import get_transform
from whar_datasets.utils.loading import load_window
from whar_datasets.utils.logging import logger


def prepare_windows_seq(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    window_df: pd.DataFrame,
    windows_dir: Path,
) -> Dict[str, List[np.ndarray]]:
    logger.info("Normalizing and transforming windows")

    normalize = get_normalize(cfg, norm_params)
    transform = get_transform(cfg)

    def prepare(window_id: str) -> Tuple[str, List[np.ndarray]]:
        window = load_window(windows_dir, window_id)
        normalized = normalize(window).values
        transformed = transform(normalized)
        return window_id, [normalized, *transformed]

    loop = tqdm(window_df["window_id"])
    loop.set_description("Normalizing and transforming windows")

    prepared: Dict[str, List[np.ndarray]] = {
        window_id: values for window_id, values in map(prepare, loop)
    }

    return prepared


def prepare_windows_para(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    window_df: pd.DataFrame,
    windows_dir: Path,
) -> Dict[str, List[np.ndarray]]:
    logger.info("Normalizing and transforming windows")

    normalize = get_normalize(cfg, norm_params)
    transform = get_transform(cfg)

    @delayed
    def prepare_delayed(window_id: str) -> Tuple[str, List[np.ndarray]]:
        window = load_window(windows_dir, window_id)
        normalized = normalize(window).values
        transformed = transform(normalized)
        return window_id, [normalized, *transformed]

    # define processing tasks
    tasks = [
        prepare_delayed(window_id)
        for window_id in [str(x) for x in window_df["window_id"]]
    ]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    tuples = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    prepared: Dict[str, List[np.ndarray]] = {
        window_id: values for window_id, values in tuples
    }

    return prepared
