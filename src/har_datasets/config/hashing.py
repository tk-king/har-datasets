import hashlib
from har_datasets.config.config import HARConfig


def create_cfg_hash(cfg: HARConfig) -> str:
    print("Creating config hash...")

    # convert to json
    cfg_json = cfg.model_dump_json()

    # create hash from json
    hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    return hash
