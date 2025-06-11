from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd

from har_datasets.dataset.har_dataset import HARDataset
from har_datasets.parsing.parsing import parse_uci_har
from har_datasets.config.configs import get_config_uci_har
from har_datasets.config.schema import Config


class HAR_DATASET_ID(Enum):
    UCI_HAR = "uci_har"
    WISDM = "wisdm"


@dataclass
class HARDatasetInfo:
    id: HAR_DATASET_ID
    parse: Callable[[str], pd.DataFrame]
    cfg: Config


HAR_DATASETS_DICT = {
    HAR_DATASET_ID.UCI_HAR: HARDatasetInfo(
        id=HAR_DATASET_ID.UCI_HAR,
        parse=parse_uci_har,
        cfg=get_config_uci_har(),
    ),
}


def get_har_dataset(
    dataset_id: HAR_DATASET_ID, cfg: Config | None = None
) -> HARDataset:
    info = HAR_DATASETS_DICT[dataset_id]
    dataset = HARDataset(cfg=cfg if cfg is not None else info.cfg, parse=info.parse)
    return dataset
