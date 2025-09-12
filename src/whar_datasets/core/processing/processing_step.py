from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path
from typing import Any, List, Set

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.logging import logger


class ProcessingStep(ABC):
    def __init__(
        self,
        cfg: WHARConfig,
        hash_dir: Path,
        dependent_on: List["ProcessingStep"] = [],
    ):
        self.cfg = cfg
        self.hash_dir = hash_dir
        self.hash_name: str
        self.relevant_cfg_keys: Set[str] = set()
        self.relevant_values: List[str] = []
        self.dependent_on = dependent_on

    def compute_hash(self) -> str:
        # hash based on relevant part of own config
        sub_cfg_json = self.cfg.model_dump_json(include=self.relevant_cfg_keys)
        base_hash = hashlib.sha256(sub_cfg_json.encode("utf-8")).hexdigest()

        # collect dependency hashes
        dep_hashes = [dep.load_hash() for dep in self.dependent_on]

        # Combine own hash + dependency hashes
        combined_str = base_hash + "".join(dep_hashes) + "".join(self.relevant_values)
        final_hash = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

        return final_hash

    def save_hash(self, hash: str):
        os.makedirs(self.hash_dir, exist_ok=True)

        hash_path = self.hash_dir / f"{self.hash_name}.txt"
        with open(hash_path, "w") as f:
            f.write(hash)

    def load_hash(self) -> str:
        hash_path = self.hash_dir / f"{self.hash_name}.txt"

        if not hash_path.exists():
            return ""

        with open(hash_path, "r") as f:
            return f.read().strip()

    def check_hash_uptodate(self) -> bool:
        logger.info(f"Checking hash for {self.__class__.__name__}")

        old_hash = self.load_hash()
        new_hash = self.compute_hash()

        return old_hash == new_hash

    def run(self, base: Any | None, force_recompute: bool) -> Any | None:
        logger.info(f"Running {self.__class__.__name__}")

        if self.check_hash_uptodate() and not force_recompute:
            logger.info("Hash is up to date")

            return self.load_results(base)

        if not self.check_initial_format(base):
            raise ValueError("Initial format check failed")

        results = self.compute_results(base)
        self.save_results(results)

        hash = self.compute_hash()
        self.save_hash(hash)

        return results

    @abstractmethod
    def check_initial_format(self, base: Any | None) -> bool:
        pass

    @abstractmethod
    def compute_results(self, base: Any | None) -> Any:
        pass

    @abstractmethod
    def save_results(self, results: Any) -> None:
        pass

    @abstractmethod
    def load_results(self, base: Any | None) -> Any:
        pass
