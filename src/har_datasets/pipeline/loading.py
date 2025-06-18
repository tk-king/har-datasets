from collections import defaultdict
import os
import tarfile
from typing import Callable, Tuple
import zipfile
import pandas as pd
from pandas import read_csv
import requests
# import dask.dataframe as dd


def load_df(
    url: str,
    datasets_dir: str,
    csv_file: str,
    parse: Callable[[str], pd.DataFrame],
    override_csv: bool = False,
) -> Tuple[pd.DataFrame, str]:
    print("Loading data...")

    # download dataset and unzip if necessary
    file_path, dataset_dir = download(url, datasets_dir)
    dataset_dir = extract(file_path, dataset_dir)

    # path to csv
    csv_path = os.path.join(dataset_dir, csv_file)

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
        df = parse(dataset_dir)
        df.to_csv(csv_path, index=True)

    return df, dataset_dir


def download(url: str, datasets_dir: str) -> Tuple[str, str]:
    os.makedirs(datasets_dir, exist_ok=True)

    # Create gitignore file if it doesn't exist
    gitignore_path = os.path.join(datasets_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("*")

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
    if not os.path.exists(file_path):
        return save_dir

    # extract zip or tar file
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as tar:
            print(f"Extracting {file_path} to {save_dir}")
            tar.extractall(save_dir)
    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zip:
            print(f"Extracting {file_path} to {save_dir}")
            zip.extractall(save_dir)
    else:
        return save_dir

    # recursively extract inner zip or tar files
    for root, _, files in os.walk(save_dir):
        for file in files:
            path = os.path.join(root, file)
            if tarfile.is_tarfile(path) or zipfile.is_zipfile(path):
                inner_file_path = os.path.join(root, file)
                inner_extract_dir = os.path.join(root, file.rsplit(".", 1)[0])
                extract(inner_file_path, inner_extract_dir)

    # clean up
    os.remove(file_path)

    return save_dir


# def load_df_dask(
#     url: str,
#     datasets_dir: str,
#     csv_file: str,
#     parse: Callable[[str], dd.DataFrame],
#     override_csv: bool = False,
# ) -> dd.DataFrame:
#     # Download and extract dataset
#     file_path, dir = download(url, datasets_dir)
#     dir = extract(file_path, dir)

#     # Path to Parquet directory
#     parquet_dir = os.path.join(dir, "parquet/")

#     # If Parquet exists and override is False, load it
#     if os.path.exists(parquet_dir) and not override_csv:
#         df = dd.read_parquet(
#             parquet_dir, engine="pyarrow", dtype_backend="numpy_nullable"
#         )

#         # set types
#         df["subject_id"] = df["subject_id"].astype("Int32")
#         df["activity_id"] = df["activity_id"].astype("Int32")
#         df["session_id"] = df["session_id"].astype("Int32")
#         df["activity_name"] = df["activity_name"].astype("string")
#         df["timestamp"] = dd.to_datetime(df["timestamp"], unit="ns")

#     else:
#         os.makedirs(parquet_dir, exist_ok=True)

#         # Parse using provided function
#         df = parse(dir)

#         # convert to dd
#         df = dd.from_pandas(df, npartitions=1)

#         # Save to partitioned Parquet by subject_id
#         df.to_parquet(
#             parquet_dir,
#             engine="pyarrow",
#             write_index=False,
#             partition_on=["subject_id"],
#         )

#     return df
