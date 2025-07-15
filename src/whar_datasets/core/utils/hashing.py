import hashlib
import json
from typing import Dict, Tuple
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


def create_norm_params_hash(
    norm_params: Tuple[Dict[str, float], Dict[str, float]] | None,
) -> str:
    print("Creating normalization parameters hash...")

    # convert to json
    norm_params_json = json.dumps(norm_params, sort_keys=True)

    # create hash from json
    hash = hashlib.sha256(
        norm_params_json.encode("utf-8") if norm_params_json else b""
    ).hexdigest()

    return hash


def load_cfg_hash(hashes_dir: str) -> str:
    print("Loading config hash...")

    # define cfg hash file path
    hash_path = os.path.join(hashes_dir, "cfg_hash.txt")

    # return cfg hash
    if not os.path.exists(hash_path):
        return ""
    else:
        with open(os.path.join(hashes_dir, "cfg_hash.txt"), "r") as f:
            return f.read()


def load_norm_params_hash(hashes_dir: str) -> str:
    print("Loading normalization parameters hash...")

    # define cfg hash file path
    hash_path = os.path.join(hashes_dir, "norm_params_hash.txt")

    # return cfg hash
    if not os.path.exists(hash_path):
        return ""
    else:
        with open(os.path.join(hashes_dir, "norm_params_hash.txt"), "r") as f:
            return f.read()
