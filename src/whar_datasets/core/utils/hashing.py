import hashlib
from whar_datasets.core.config import WHARConfig
import os


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


def load_cfg_hash(cache_dir: str) -> str:
    print("Loading config hash...")

    cache_path = os.path.join(cache_dir, "cfg_hash.txt")

    if not os.path.exists(cache_path):
        return ""
    else:
        with open(os.path.join(cache_dir, "cfg_hash.txt"), "r") as f:
            return f.read()
