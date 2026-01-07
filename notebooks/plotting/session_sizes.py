# %%
import os
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from whar_datasets.core.utils.loading import load_session_df
from whar_datasets.config.getter import WHARDatasetID, get_dataset_cfg
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

dict: Dict[str, List[float]] = {
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

for dataset_id in dataset_ids:
    cfg = get_dataset_cfg(dataset_id, datasets_dir="./notebooks/datasets/")
    datasets_dir = cfg.common.datasets_dir
    dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)
    cache_dir = os.path.join(dataset_dir, "cache/")
    sessions_dir = os.path.join(cache_dir, "sessions/")
    session_df = load_session_df(cache_dir)

    loop = tqdm(session_df["session_id"])
    loop.set_description(dataset_id.value)

    for session_id in loop:
        assert isinstance(session_id, int)

        # get and save session
        session_path = os.path.join(sessions_dir, f"session_{session_id}.csv")
        session_df = pd.read_csv(session_path)

        dict[dataset_id.value].append(session_df.shape[0])


with open("session_sizes.json", "w") as f:
    json.dump(dict, f)


# %%
