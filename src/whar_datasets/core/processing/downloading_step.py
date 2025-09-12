import os
from pathlib import Path
import tarfile
from typing import Any, Set
import zipfile

import requests

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.processing.processing_step import ProcessingStep
from whar_datasets.core.utils.logging import logger


class DownloadingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        hash_dir: Path,
        datasets_dir: Path,
        dataset_dir: Path,
    ):
        super().__init__(cfg, hash_dir)

        self.datasets_dir = datasets_dir
        self.dataset_dir = dataset_dir

        self.hash_name: str = "download_hash"
        self.relevant_cfg_keys: Set[str] = {
            "dataset_id",
            "datasets_dir",
            "download_url",
        }

    def check_initial_format(self, base: None) -> bool:
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Create gitignore file if it doesn't exist
        gitignore_path = self.datasets_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write("*")

        return True

    def compute_results(self, base: Any | None) -> Any:
        logger.info(f"Downloading {self.cfg.download_url} to {self.dataset_dir}")

        # Use filename to define file path
        filename = self.cfg.download_url.split("/")[-1]
        file_path = self.dataset_dir / filename

        # download file from url
        response = requests.get(self.cfg.download_url)
        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path

    def save_results(self, results: Any) -> None:
        logger.info(f"Extracting {self.cfg.dataset_id} dataset")

        file_path = results

        def extract_dir(inner_file_path: Path, inner_extract_dir: Path) -> None:
            if tarfile.is_tarfile(inner_file_path):
                with tarfile.open(inner_file_path) as tar:
                    logger.info(f"Extracting {inner_file_path} to {inner_extract_dir}")
                    inner_extract_dir.mkdir(parents=True, exist_ok=True)
                    tar.extractall(inner_extract_dir)
            elif zipfile.is_zipfile(inner_file_path):
                with zipfile.ZipFile(inner_file_path) as zip:
                    logger.info(f"Extracting {inner_file_path} to {inner_extract_dir}")
                    inner_extract_dir.mkdir(parents=True, exist_ok=True)
                    zip.extractall(inner_extract_dir)
            else:
                return

            # recursively check extracted files
            for root, _, files in os.walk(inner_extract_dir):
                for file in files:
                    nested_path = Path(root) / file
                    if tarfile.is_tarfile(nested_path) or zipfile.is_zipfile(
                        nested_path
                    ):
                        nested_extract_dir = nested_path.with_suffix(
                            ""
                        )  # remove extension
                        extract_dir(nested_path, nested_extract_dir)
                        os.remove(nested_path)  # cleanup after extraction

        extract_dir(file_path, self.dataset_dir)

    def load_results(self) -> Any | None:
        return None
