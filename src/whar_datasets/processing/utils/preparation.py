from pathlib import Path
from typing import Dict, List, Tuple

import dask.dataframe as dd
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
    logger.info("Normalizing and transforming windows (parallelized)")

    normalize = get_normalize(cfg, norm_params)
    transform = get_transform(cfg)

    relevant_ids = set(window_df["window_id"])

    # Read parquet with dask to handle partitions efficiently
    ddf = dd.read_parquet(windows_dir / "windows.parquet", engine="pyarrow")

    def process_partition(df: pd.DataFrame) -> List[Tuple[str, List[np.ndarray]]]:
        results: List[Tuple[str, List[np.ndarray]]] = []
        if df.empty:
            return results

        # Check if any window in this partition is relevant
        # This avoids processing partitions that don't contain any relevant windows
        if "window_id" not in df.columns:
            return results

        # Filter for relevant windows in this partition
        mask = df["window_id"].isin(relevant_ids)
        if not mask.any():
            return results

        subset = df[mask]

        # Group by window_id and process
        for window_id, group in subset.groupby("window_id"):
            # Drop window_id column to match load_window behavior
            window_data = group.drop(columns=["window_id"]).reset_index(drop=True)

            normalized = normalize(window_data).values
            transformed = transform(normalized)
            results.append((str(window_id), [normalized, *transformed]))

        return results

    # Create delayed tasks for each partition
    delayed_partitions = ddf.to_delayed()

    @delayed
    def process_delayed(partition):
        return process_partition(partition)

    tasks = [process_delayed(part) for part in delayed_partitions]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    results_list = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    # Flatten results
    prepared: Dict[str, List[np.ndarray]] = {}
    for partition_results in results_list:
        for window_id, data in partition_results:
            prepared[window_id] = data

    return prepared
