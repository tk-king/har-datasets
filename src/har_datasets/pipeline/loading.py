from collections import defaultdict
import os
import tarfile
from typing import Callable
import zipfile
import pandas as pd
from pandas import read_csv
import requests
# import dask.dataframe as dd


def get_df(
    datasets_dir: str,
    dataset_id: str,
    dataset_url: str,
    parse: Callable[[str], pd.DataFrame],
    override_csv: bool = False,
) -> pd.DataFrame:
    print("Loading data...")

    dataset_dir = os.path.join(datasets_dir, dataset_id)
    csv_path = os.path.join(dataset_dir, dataset_id + ".csv")

    # download dataset and unzip if necessary
    file_path = download_dataset(datasets_dir, dataset_dir, dataset_url)
    extract_dataset(file_path, dataset_dir)

    # if file exists, load it, else parse from dataset and save
    if os.path.exists(csv_path) and not override_csv:
        # set types for cols, default to float
        # read csv while setting types and parsing timestamp
        df = read_csv(
            csv_path,
            dtype=defaultdict(
                lambda: float,
                **{
                    "subject_id": "int32",
                    "activity_name": "str",
                    "activity_id": "int32",
                    "session_id": "int32",
                },
            ),
            parse_dates=["timestamp"],
        )
    else:
        df = parse(dataset_dir)
        df.to_csv(csv_path, index=True)

    return df


def download_dataset(datasets_dir: str, dataset_dir: str, dataset_url: str) -> str:
    os.makedirs(datasets_dir, exist_ok=True)

    # Create gitignore file if it doesn't exist
    gitignore_path = os.path.join(datasets_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("*")

    filename = dataset_url.split("/")[-1]
    file_path = os.path.join(datasets_dir, filename)

    # If zip or dir already exists, do nothing
    if os.path.exists(file_path) or os.path.exists(dataset_dir):
        return file_path

    # else download file from url
    else:
        print(f"Downloading {dataset_url} to {datasets_dir}")
        response = requests.get(dataset_url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    return file_path


def extract_dataset(file_path: str, dataset_dir: str):
    # check if extract is necessary
    if not os.path.exists(file_path):
        return

    # extract zip or tar file
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as tar:
            print(f"Extracting {file_path} to {dataset_dir}")
            tar.extractall(dataset_dir)
    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zip:
            print(f"Extracting {file_path} to {dataset_dir}")
            zip.extractall(dataset_dir)
    else:
        return

    # recursively extract inner zip or tar files
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            path = os.path.join(root, file)
            if tarfile.is_tarfile(path) or zipfile.is_zipfile(path):
                inner_file_path = os.path.join(root, file)
                inner_extract_dir = os.path.join(root, file.rsplit(".", 1)[0])
                extract_dataset(inner_file_path, inner_extract_dir)

    # clean up
    os.remove(file_path)


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
