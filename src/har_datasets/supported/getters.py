from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd

from har_datasets.dataset.har_dataset import HARDataset
from har_datasets.supported.parsing import parse_uci_har
from har_datasets.supported.configs import get_config_uci_har
from har_datasets.config.config import HARConfig


class HAR_DATASET_ID(Enum):
    UCI_HAR = "uci_har"
    WISDM = "wisdm"


@dataclass
class HARDatasetInfo:
    id: HAR_DATASET_ID
    parse: Callable[[str], pd.DataFrame]
    cfg: HARConfig


HAR_DATASETS_DICT = {
    HAR_DATASET_ID.UCI_HAR: HARDatasetInfo(
        id=HAR_DATASET_ID.UCI_HAR,
        parse=parse_uci_har,
        cfg=get_config_uci_har(),
    ),
}


def get_har_dataset(
    dataset_id: HAR_DATASET_ID, cfg: HARConfig | None = None
) -> HARDataset:
    # get dataset info
    info = HAR_DATASETS_DICT[dataset_id]

    # if cfg is None, use default config for specific dataset
    dataset = HARDataset(cfg=cfg if cfg is not None else info.cfg, parse=info.parse)

    return dataset
