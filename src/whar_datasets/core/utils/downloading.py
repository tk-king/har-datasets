import os
import tarfile
import zipfile
import requests


def download(datasets_dir: str, dataset_dir: str, dataset_url: str) -> str:
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


def extract(file_path: str, dataset_dir: str):
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
                extract(inner_file_path, inner_extract_dir)

    # clean up
    os.remove(file_path)
