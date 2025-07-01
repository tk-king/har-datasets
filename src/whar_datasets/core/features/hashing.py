import hashlib
from whar_datasets.core.config import WHARConfig


def create_cfg_hash(cfg: WHARConfig) -> str:
    print("Creating config hash...")

    # copy config to not modify original
    cfg = cfg.model_copy(deep=True)

    # ignore training for hashing
    cfg.dataset.training = None  # type: ignore

    # convert to json
    cfg_json = cfg.model_dump_json()

    # create hash from json
    hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    return hash
