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
    cfg.batch_size = None  # type: ignore
    cfg.learning_rate = None  # type: ignore
    cfg.num_epochs = None  # type: ignore
    cfg.seed = None  # type: ignore
    cfg.in_memory = None  # type: ignore
    cfg.given_train_test_subj_ids = None  # type: ignore
    cfg.subj_cross_val_split_groups = None  # type: ignore
    cfg.val_percentage = None  # type: ignore
    cfg.normalization = None  # type: ignore

    # convert to json
    cfg_json = cfg.model_dump_json()

    # create hash from json
    hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    return hash


def create_norm_params_hash(
    cfg_hash: str,
    norm_params: Tuple[Dict[str, float], Dict[str, float]] | None,
) -> str:
    print("Creating normalization parameters hash...")

    # convert to json
    norm_params_json = json.dumps(norm_params, sort_keys=True)

    # create hash from json and cfg hash
    hash = hashlib.sha256((cfg_hash + norm_params_json).encode("utf-8")).hexdigest()

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
