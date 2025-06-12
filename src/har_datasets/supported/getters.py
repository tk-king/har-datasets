from typing import Dict
from enum import Enum
from typing import Callable

import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose

from har_datasets.dataset.har_dataset import HARDataset
from har_datasets.supported.parsers import parse_uci_har
from har_datasets.config.config import HARConfig


class HAR_DATASET_ID(Enum):
    UCI_HAR = "uci_har"
    WISDM = "wisdm"


HAR_DATASETS_DICT: Dict[HAR_DATASET_ID, Callable[[str], pd.DataFrame]] = {
    HAR_DATASET_ID.UCI_HAR: parse_uci_har,
}


def get_har_dataset(
    dataset_id: HAR_DATASET_ID, cfg: HARConfig | None = None
) -> HARDataset:
    # get dataset-specific parser and config to create dataset
    parse = HAR_DATASETS_DICT[dataset_id]
    config = cfg if cfg is not None else get_har_config(dataset_id=dataset_id)
    dataset = HARDataset(cfg=config, parse=parse)

    return dataset


def get_har_config(
    dataset_id: HAR_DATASET_ID, config_dir: str = "../config"
) -> HARConfig:
    # load dataset-specific config from dir
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name="cfg", overrides=[f"dataset={dataset_id}"])
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        cfg = HARConfig(**cfg)  # type: ignore

    return cfg  # type: ignore
