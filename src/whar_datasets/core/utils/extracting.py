import os
from pathlib import Path
import tarfile
import zipfile


def extract_dir(inner_file_path: Path, inner_extract_dir: Path) -> None:
    # extract inner file
    if tarfile.is_tarfile(inner_file_path):
        with tarfile.open(inner_file_path) as tar:
            inner_extract_dir.mkdir(parents=True, exist_ok=True)
            tar.extractall(inner_extract_dir)
    elif zipfile.is_zipfile(inner_file_path):
        with zipfile.ZipFile(inner_file_path) as zip:
            inner_extract_dir.mkdir(parents=True, exist_ok=True)
            zip.extractall(inner_extract_dir)
    else:
        return None

    # recursively check extracted files
    for root, _, files in os.walk(inner_extract_dir):
        for file in files:
            nested_path = Path(root) / file
            if tarfile.is_tarfile(nested_path) or zipfile.is_zipfile(nested_path):
                nested_extract_dir = nested_path.with_suffix("")  # remove extension
                extract_dir(nested_path, nested_extract_dir)
                os.remove(nested_path)  # cleanup after extraction
