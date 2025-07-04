import pandas as pd
from collections import Counter


def compute_class_weights(
    session_index: pd.DataFrame, window_index: pd.DataFrame
) -> dict:
    # Merge to assign activity_id to each window
    merged = window_index.merge(session_index, on="session_id", how="left")

    print(f"activity_ids: {merged['activity_id'].unique()}")

    # Extract activity_id per window
    window_labels = merged["activity_id"].tolist()

    # Count occurrences
    label_counts = Counter(window_labels)
    total = sum(label_counts.values())

    # Inverse frequency
    class_weights = {label: total / count for label, count in label_counts.items()}

    # Normalize to mean = 1 (optional)
    mean_weight = sum(class_weights.values()) / len(class_weights)
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}

    print("num of class weights: ", len(class_weights))

    return class_weights
