from collections import defaultdict
import os
import tarfile
from typing import Callable, Tuple
import pandas as pd
from pandas import read_csv
import requests


def load_df(
    url: str,
    datasets_dir: str,
    csv_file: str,
    parse: Callable[[str], pd.DataFrame],
    override_csv: bool = False,
) -> pd.DataFrame:
    # download dataset
    file_path, dir = download(url, datasets_dir)

    # unzip if necessary
    dir = extract(file_path, dir)

    # path to csv
    csv_path = os.path.join(dir, csv_file)

    # if file exists, load it, else parse from dataset and save
    if os.path.exists(csv_path) and not override_csv:
        # set types for cols, default to float
        types: dict = defaultdict(
            lambda: float,
            **{
                "subject_id": "int32",
                "activity_name": "str",
                "activity_id": "int32",
                "session_id": "int32",
            },
        )

        # read csv while setting types and parsing timestamp
        df = read_csv(
            csv_path,
            dtype=types,
            parse_dates=["timestamp"],
        )
    else:
        df = parse(dir)
        df.to_csv(csv_path, index=True)

    return df


def download(url: str, datasets_dir: str) -> Tuple[str, str]:
    os.makedirs(datasets_dir, exist_ok=True)

    filename = url.split("/")[-1]
    file_path = os.path.join(datasets_dir, filename)
    dir = os.path.join(datasets_dir, filename.rsplit(".", 1)[0])

    # If zip or dir already exists, do nothing
    if os.path.exists(file_path) or os.path.exists(dir):
        return file_path, dir

    # else download file from url
    else:
        print(f"Downloading {url} to {datasets_dir}")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    return file_path, dir


def extract(file_path: str, save_dir: str):
    # check if extract is necessary
    if not os.path.exists(file_path) or not tarfile.is_tarfile(file_path):
        return save_dir

    # extract
    with tarfile.open(file_path) as tar:
        print(f"Extracting {file_path} to {save_dir}")
        tar.extractall(save_dir)

    # recursively extract inner .tar files
    for root, _, files in os.walk(save_dir):
        for file in files:
            if tarfile.is_tarfile(os.path.join(root, file)):
                inner_zip_path = os.path.join(root, file)
                inner_extract_dir = os.path.join(root, file.rsplit(".", 1)[0])
                extract(inner_zip_path, inner_extract_dir)

    # clean up
    os.remove(file_path)

    return save_dir
