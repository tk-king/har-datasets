import os
import tarfile
import zipfile
from pathlib import Path


def extract(file_path: Path, extract_dir: Path) -> None:
    extracted = False

    # extract depending on file type
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as tar:
            extract_dir.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_dir)
        extracted = True
    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zipf:
            extract_dir.mkdir(parents=True, exist_ok=True)
            zipf.extractall(extract_dir)
        extracted = True

    # ONLY clean up if we actually extracted something
    if extracted:
        file_path.unlink()

    # recursively extract nested archives
    for root, _, files in os.walk(extract_dir):
        for file in files:
            nested_path = Path(root) / file
            # These checks usually effectively handle the recursion safely
            if tarfile.is_tarfile(nested_path) or zipfile.is_zipfile(nested_path):
                nested_extract_dir = nested_path.with_suffix("")
                extract(nested_path, nested_extract_dir)
