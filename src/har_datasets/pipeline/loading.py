import os
from typing import Callable, List
import pandas as pd
from pandas import read_csv
import requests
import zipfile


def load_df(
    url: str,
    datasets_dir: str,
    csv_file: str,
    parse: Callable[[str], pd.DataFrame],
    required_cols: List[str],
) -> pd.DataFrame:
    # download dataset
    dir = download(url, datasets_dir)

    # path to csv
    csv_path = os.path.join(dir, csv_file)

    # if file exists, load it, else parse from dataset and save
    if os.path.exists(csv_path):
        df = read_csv(csv_path)
    else:
        df = parse(dir)
        df.to_csv(csv_path, index=True)

    # check that all required columns are present
    assert set(required_cols).issubset(df.columns)

    return df


def download(url: str, datasets_dir: str) -> str:
    os.makedirs(datasets_dir, exist_ok=True)

    filename = url.split("/")[-1]
    file_path = os.path.join(datasets_dir, filename)
    zip_dir = os.path.join(datasets_dir, filename.rsplit(".", 1)[0])

    # If file already exists, do nothing
    if os.path.exists(file_path) or os.path.exists(zip_dir):
        print(f"File {filename} already exists in {datasets_dir}")

    # else download file from url
    else:
        print(f"Downloading {url} to {datasets_dir}")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    # Unzip .zip files
    if filename.endswith(".zip"):
        unzip(file_path, zip_dir)

    # Recursively remove .zip files
    for root, _, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith(".zip"):
                os.remove(os.path.join(root, file))

    return zip_dir


def unzip(zip_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Directory {save_dir} already exists and is not empty")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"Unzipped {zip_path} to {save_dir}")

    # Recursively unzip inner .zip files
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".zip"):
                inner_zip_path = os.path.join(root, file)
                inner_extract_dir = os.path.join(root, file.rsplit(".", 1)[0])
                unzip(inner_zip_path, inner_extract_dir)
