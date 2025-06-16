from enum import Enum
from typing import Callable, Dict, Tuple
import pandas as pd
from omegaconf import OmegaConf
from hydra import initialize, compose

from har_datasets.config.config import HARConfig
from har_datasets.parsers.parse_uci_har import parse_uci_har
from har_datasets.parsers.parse_wisdm_19 import (
    parse_wisdm_19_phone,
    parse_wisdm_19_watch,
)
from har_datasets.parsers.parse_wisdm_12 import parse_wisdm_12


class DatasetId(Enum):
    UCI_HAR = "uci_har"
    WISDM_19_PHONE = "wisdm_19_phone"
    WISDM_19_WATCH = "wisdm_19_watch"
    WISDM_12 = "wisdm_12"


HAR_DATASETS_DICT: Dict[DatasetId, Callable[[str], pd.DataFrame]] = {
    DatasetId.UCI_HAR: parse_uci_har,
    DatasetId.WISDM_19_PHONE: parse_wisdm_19_phone,
    DatasetId.WISDM_19_WATCH: parse_wisdm_19_watch,
    DatasetId.WISDM_12: parse_wisdm_12,
}


def get_har_dataset_cfg_and_parser(
    dataset_id: DatasetId, config_dir: str = "../../../config"
) -> Tuple[HARConfig, Callable[[str], pd.DataFrame]]:
    # load dataset-specific config from dir
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name="cfg", overrides=[f"dataset={dataset_id.value}"])
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        cfg = HARConfig(**cfg)  # type: ignore
        assert isinstance(cfg, HARConfig)

    # get parser corresponding to dataset
    parse = HAR_DATASETS_DICT[dataset_id]

    return cfg, parse
