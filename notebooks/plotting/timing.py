# %%
import os
import time
from typing import Dict, List
from whar_datasets.core.preprocessing import (
    process_sessions_parallely,
    process_sessions_sequentially,
)
from whar_datasets.core.utils.loading import load_session_metadata
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg
import json


dataset_ids = [
    WHARDatasetID.OPPORTUNITY,
    WHARDatasetID.UCI_HAR,
    WHARDatasetID.WISDM_12,
    WHARDatasetID.PAMAP2,
    WHARDatasetID.MOTION_SENSE,
    WHARDatasetID.MHEALTH,
    WHARDatasetID.DSADS,
    WHARDatasetID.KU_HAR,
    WHARDatasetID.DAPHNET,
    WHARDatasetID.HAR_SENSE,
]

parallel_times_dict: Dict[str, List[float]] = {
    WHARDatasetID.UCI_HAR.value: [],
    WHARDatasetID.WISDM_12.value: [],
    WHARDatasetID.PAMAP2.value: [],
    WHARDatasetID.MOTION_SENSE.value: [],
    WHARDatasetID.OPPORTUNITY.value: [],
    WHARDatasetID.MHEALTH.value: [],
    WHARDatasetID.DSADS.value: [],
    WHARDatasetID.KU_HAR.value: [],
    WHARDatasetID.DAPHNET.value: [],
    WHARDatasetID.HAR_SENSE.value: [],
}

sequential_times_dict: Dict[str, List[float]] = {
    WHARDatasetID.UCI_HAR.value: [],
    WHARDatasetID.WISDM_12.value: [],
    WHARDatasetID.PAMAP2.value: [],
    WHARDatasetID.MOTION_SENSE.value: [],
    WHARDatasetID.OPPORTUNITY.value: [],
    WHARDatasetID.MHEALTH.value: [],
    WHARDatasetID.DSADS.value: [],
    WHARDatasetID.KU_HAR.value: [],
    WHARDatasetID.DAPHNET.value: [],
    WHARDatasetID.HAR_SENSE.value: [],
}

num_samples = 3

if __name__ == "__main__":
    for dataset_id in dataset_ids:
        cfg = get_whar_cfg(dataset_id, datasets_dir="./notebooks/datasets/")
        datasets_dir = cfg.common.datasets_dir
        dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)
        cache_dir = os.path.join(dataset_dir, "cache/")
        sessions_dir = os.path.join(cache_dir, "sessions/")
        session_metadata = load_session_metadata(cache_dir)

        for i in range(num_samples):
            start_time = time.time()
            process_sessions_parallely(cfg, sessions_dir, session_metadata)
            duration = time.time() - start_time
            parallel_times_dict[dataset_id.value].append(duration)
            print(f"Dataset: {dataset_id}, Sample: {i}, Duration: {duration}")

        del cfg
        del datasets_dir
        del dataset_dir
        del cache_dir
        del sessions_dir
        del session_metadata

    for dataset_id in dataset_ids:
        cfg = get_whar_cfg(dataset_id, datasets_dir="./notebooks/datasets/")
        datasets_dir = cfg.common.datasets_dir
        dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)
        cache_dir = os.path.join(dataset_dir, "cache/")
        sessions_dir = os.path.join(cache_dir, "sessions/")
        session_metadata = load_session_metadata(cache_dir)

        for i in range(num_samples):
            start_time = time.time()
            process_sessions_sequentially(cfg, sessions_dir, session_metadata)
            duration = time.time() - start_time
            sequential_times_dict[dataset_id.value].append(duration)
            print(f"Dataset: {dataset_id}, Sample: {i}, Duration: {duration}")

        del cfg
        del datasets_dir
        del dataset_dir
        del cache_dir
        del sessions_dir
        del session_metadata

    with open("parallel_times.json", "w") as f:
        json.dump(parallel_times_dict, f)

    with open("sequential_times.json", "w") as f:
        json.dump(sequential_times_dict, f)


# %%
