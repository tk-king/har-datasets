import requests
import os
import zipfile


def download(url: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    filename = url.split("/")[-1]
    file_path = os.path.join(save_dir, filename)

    if os.path.exists(file_path):
        print(f"File {filename} already exists in {save_dir}")
    else:
        print(f"Downloading {url} to {save_dir}")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    if filename.endswith(".zip"):
        unzip(file_path, os.path.join(save_dir, filename.rsplit(".", 1)[0]))

    # Recursively remove .zip files
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".zip"):
                os.remove(os.path.join(root, file))


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
