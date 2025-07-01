from enum import Enum
from typing import Callable, Dict, Tuple
import pandas as pd

from whar_datasets.core.config import WHARConfig
from whar_datasets.support.configs.cfg_ku_har import cfg_ku_har
from whar_datasets.support.configs.cfg_dsads import cfg_dsads
from whar_datasets.support.configs.cfg_mhealth import cfg_mhealth
from whar_datasets.support.configs.cfg_opportunity import cfg_opportunity
from whar_datasets.support.configs.cfg_pamap2 import cfg_pamap2
from whar_datasets.support.configs.cfg_wisdm_12 import cfg_wisdm_12
from whar_datasets.support.configs.cfg_uci_har import cfg_uci_har
from whar_datasets.support.configs.cfg_motion_sense import cfg_motion_sense
from whar_datasets.support.configs.cfg_daphnet import cfg_daphnet
from whar_datasets.support.configs.cfg_har_sense import cfg_har_sense
from whar_datasets.support.parsers.parse_har_sense import parse_har_sense
from whar_datasets.support.parsers.parse_daphnet import parse_daphnet
from whar_datasets.support.parsers.parse_ku_har import parse_ku_har
from whar_datasets.support.parsers.parse_dsads import parse_dsads
from whar_datasets.support.parsers.parse_mhealth import parse_mhealth
from whar_datasets.support.parsers.parse_opportunity import parse_opportunity
from whar_datasets.support.parsers.parse_pamap2 import parse_pamap2
from whar_datasets.support.parsers.parse_uci_har import parse_uci_har
from whar_datasets.support.parsers.parse_wisdm_12 import parse_wisdm_12
from whar_datasets.support.parsers.parse_motion_sense import parse_motion_sense


class WHARDatasetID(Enum):
    UCI_HAR = "uci_har"
    WISDM_12 = "wisdm_12"
    PAMAP2 = "pamap2"
    MOTION_SENSE = "motion_sense"
    OPPORTUNITY = "opportunity"
    MHEALTH = "mhealth"
    DSADS = "dsads"
    KU_HAR = "ku_har"
    DAPHNET = "daphnet"
    HAR_SENSE = "har_sense"


har_dataset_dict: Dict[
    WHARDatasetID, Tuple[WHARConfig, Callable[[str, str], pd.DataFrame]]
] = {
    WHARDatasetID.UCI_HAR: (cfg_uci_har, parse_uci_har),
    WHARDatasetID.WISDM_12: (cfg_wisdm_12, parse_wisdm_12),
    WHARDatasetID.PAMAP2: (cfg_pamap2, parse_pamap2),
    WHARDatasetID.MOTION_SENSE: (cfg_motion_sense, parse_motion_sense),
    WHARDatasetID.OPPORTUNITY: (cfg_opportunity, parse_opportunity),
    WHARDatasetID.MHEALTH: (cfg_mhealth, parse_mhealth),
    WHARDatasetID.DSADS: (cfg_dsads, parse_dsads),
    WHARDatasetID.KU_HAR: (cfg_ku_har, parse_ku_har),
    WHARDatasetID.DAPHNET: (cfg_daphnet, parse_daphnet),
    WHARDatasetID.HAR_SENSE: (cfg_har_sense, parse_har_sense),
}


def get_cfg_and_parser(
    dataset_id: WHARDatasetID, datasets_dir: str = "./datasets"
) -> Tuple[WHARConfig, Callable[[str, str], pd.DataFrame]]:
    # load dataset-specific config and parser
    cfg, parse = har_dataset_dict[dataset_id]

    # override datasets dir
    cfg.common.datasets_dir = datasets_dir

    return cfg, parse
