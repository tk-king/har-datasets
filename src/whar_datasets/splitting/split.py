from dataclasses import dataclass
from typing import List


@dataclass
class Split:
    identifier: str
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
