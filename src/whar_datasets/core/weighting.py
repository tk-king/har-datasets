import pandas as pd
from collections import Counter


def compute_class_weights(session_index: pd.DataFrame) -> dict:
    # Get label for each session
    session_labels = session_index["activity_id"].tolist()

    # Count occurrences
    label_counts = Counter(session_labels)
    total = sum(label_counts.values())

    # Inverse frequency
    class_weights = {label: total / count for label, count in label_counts.items()}

    # Normalize to mean = 1 (optional)
    mean_weight = sum(class_weights.values()) / len(class_weights)
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}

    return class_weights
