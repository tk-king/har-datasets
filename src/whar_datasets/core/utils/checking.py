import os

from whar_datasets.core.utils.hashing import load_cfg_hash


def check_download(dataset_dir: str) -> bool:
    print("Checking download...")

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found at '{dataset_dir}'.")
        return False

    print("Download exists.")
    return True


def check_sessions(cache_dir: str, sessions_dir: str) -> bool:
    print("Checking sessions...")

    if not os.path.exists(sessions_dir):
        print(f"Sessions directory not found at '{sessions_dir}'.")
        return False

    if len(os.listdir(sessions_dir)) == 0:
        print(f"Sessions directory '{sessions_dir}' is empty.")
        return False

    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")
    if not os.path.exists(session_metadata_path):
        print(f"Session index file not found at '{session_metadata_path}'.")
        return False

    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")
    if not os.path.exists(activity_metadata_path):
        print(f"Activity index file not found at '{activity_metadata_path}'.")
        return False

    print("Sessions exist.")
    return True


def check_windowing(
    cache_dir: str, windows_dir: str, hashes_dir: str, cfg_hash: str
) -> bool:
    print("Checking windowing...")

    if not os.path.exists(windows_dir):
        print(f"Windows directory not found at '{windows_dir}'.")
        return False

    if len(os.listdir(windows_dir)) == 0:
        print(f"Windows directory '{windows_dir}' is empty.")
        return False

    window_metadata_path = os.path.join(cache_dir, "window_metadata.parquet")

    if not os.path.exists(window_metadata_path):
        print(f"Window index file not found at '{window_metadata_path}'.")
        return False

    current_hash = load_cfg_hash(hashes_dir)

    if cfg_hash != current_hash:
        print("Config hash mismatch.")
        return False

    print("Windowing exists.")
    return True
