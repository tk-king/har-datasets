from enum import Enum
from typing import Callable, Dict, Tuple
import pandas as pd
# from omegaconf import OmegaConf
# from hydra import initialize, compose

from har_datasets.config.config import HARConfig
from har_datasets.supported.configs.cfg_mhealth import cfg_mhealth
from har_datasets.supported.configs.cfg_opportunity import cfg_opportunity
from har_datasets.supported.configs.cfg_pamap2 import cfg_pamap2
from har_datasets.supported.configs.cfg_wisdm_12 import cfg_wisdm_12
from har_datasets.supported.configs.cfg_uci_har import cfg_uci_har
from har_datasets.supported.configs.cfg_motion_sense import cfg_motion_sense
from har_datasets.supported.parsers.parse_mhealth import parse_mhealth
from har_datasets.supported.parsers.parse_opportunity import parse_opportunity
from har_datasets.supported.parsers.parse_pamap2 import parse_pamap2
from har_datasets.supported.parsers.parse_uci_har import parse_uci_har
from har_datasets.supported.parsers.parse_wisdm_12 import parse_wisdm_12
from har_datasets.supported.parsers.parse_motion_sense import parse_motion_sense


class DatasetId(Enum):
    UCI_HAR = "uci_har"
    WISDM_12 = "wisdm_12"
    PAMAP2 = "pamap2"
    MOTION_SENSE = "motion_sense"
    OPPORTUNITY = "opportunity"
    MHEALTH = "mhealth"


har_dataset_dict: Dict[
    DatasetId, Tuple[HARConfig, Callable[[str, str], pd.DataFrame]]
] = {
    DatasetId.UCI_HAR: (cfg_uci_har, parse_uci_har),
    DatasetId.WISDM_12: (cfg_wisdm_12, parse_wisdm_12),
    DatasetId.PAMAP2: (cfg_pamap2, parse_pamap2),
    DatasetId.MOTION_SENSE: (cfg_motion_sense, parse_motion_sense),
    DatasetId.OPPORTUNITY: (cfg_opportunity, parse_opportunity),
    DatasetId.MHEALTH: (cfg_mhealth, parse_mhealth),
}


def get_har_dataset_cfg_and_parser(
    dataset_id: DatasetId, datasets_dir: str = "./datasets"
) -> Tuple[HARConfig, Callable[[str, str], pd.DataFrame]]:
    # load dataset-specific config and parser
    cfg, parse = har_dataset_dict[dataset_id]

    # override datasets dir
    cfg.common.datasets_dir = datasets_dir

    return cfg, parse


# def get_har_dataset_cfg_and_parser(
#     dataset_id: DatasetId, config_dir: str = "../../../config"
# ) -> Tuple[HARConfig, Callable[[str], pd.DataFrame]]:
#     # load dataset-specific config from dir
#     with initialize(version_base=None, config_path=config_dir):
#         cfg = compose(config_name="cfg", overrides=[f"dataset={dataset_id.value}"])
#         cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
#         cfg = HARConfig(**cfg)  # type: ignore
#         assert isinstance(cfg, HARConfig)

#     # get parser corresponding to dataset
#     parse = HAR_DATASETS_DICT[dataset_id]

#     return cfg, parse
