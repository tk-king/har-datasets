from typing import Dict
import pandas as pd


def compute_class_weights(
    session_metadata: pd.DataFrame, window_metadata: pd.DataFrame
) -> dict:
    # Get all possible labels
    possible_labels = [int(label) for label in session_metadata["activity_id"].unique()]

    # Merge to assign activity_id to each window
    merged = window_metadata.merge(session_metadata, on="session_id", how="left")

    # Count activity_id occurrences over windows
    label_counts = merged["activity_id"].value_counts()

    # Compute inverse frequency
    total = label_counts.sum()
    class_weights: Dict[int, float] = (total / label_counts).to_dict()

    # Normalize weights to have mean = 1
    mean_weight = sum(class_weights.values()) / len(class_weights)
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}

    # Add labels not present in windows with weight -1
    for label in possible_labels:
        if label not in class_weights:
            class_weights[label] = -1

    return class_weights
