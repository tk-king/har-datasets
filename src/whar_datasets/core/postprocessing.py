from typing import Dict, List

import numpy as np
import pandas as pd
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.caching import cache_norm_params_hash, cache_samples
from whar_datasets.core.utils.hashing import (
    create_norm_params_hash,
    load_cfg_hash,
    load_norm_params_hash,
)
from whar_datasets.core.utils.normalization import (
    get_norm_params,
    normalize_windows_seq,
)
from whar_datasets.core.utils.loading import load_samples, load_windows
from whar_datasets.core.utils.transform import transform_windows_seq


def postprocess(
    cfg: WHARConfig,
    indices: List[int],
    hashes_dir: str,
    samples_dir: str,
    windows_dir: str,
    window_metadata: pd.DataFrame,
    override_cache: bool,
) -> Dict[str, List[np.ndarray]] | None:
    print("Postprocessing...")

    windows = load_windows(window_metadata, windows_dir)

    # get hash for postprocessing
    norm_params = get_norm_params(cfg, indices, window_metadata, windows)
    cfg_hash = load_cfg_hash(hashes_dir)
    norm_params_hash = create_norm_params_hash(cfg_hash, norm_params)

    # postprocess and cache if needed
    if override_cache or norm_params_hash != load_norm_params_hash(hashes_dir):
        normalized = normalize_windows_seq(cfg, norm_params, window_metadata, windows)
        transformed = transform_windows_seq(cfg, window_metadata, windows)

        # combine normalized and transformed into samples
        assert normalized.keys() == transformed.keys()
        samples: Dict[str, List[np.ndarray]] = {
            window_id: [normalized[window_id], *transformed[window_id]]
            for window_id in normalized.keys()
        }

        # cache if configured
        if cfg.cache_postprocessing:
            cache_samples(samples_dir, window_metadata, samples)
            cache_norm_params_hash(hashes_dir, norm_params_hash)

    else:
        samples = load_samples(window_metadata, samples_dir)

    return samples if cfg.in_memory else None
