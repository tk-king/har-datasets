import os
from pathlib import Path
import tarfile
import zipfile


def extract(file_path: Path, extract_dir: Path) -> None:
    # extract depending on file type
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as tar:
            extract_dir.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_dir)
    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zipf:
            extract_dir.mkdir(parents=True, exist_ok=True)
            zipf.extractall(extract_dir)

    # clean up
    file_path.unlink()

    # recursively extract nested archives
    for root, _, files in os.walk(extract_dir):
        for file in files:
            nested_path = Path(root) / file
            if tarfile.is_tarfile(nested_path) or zipfile.is_zipfile(nested_path):
                nested_extract_dir = nested_path.with_suffix("")
                extract(nested_path, nested_extract_dir)
