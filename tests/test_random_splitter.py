import unittest

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.splitter_random import RandomSplitter


def _dummy_cfg(seed: int = 0) -> WHARConfig:
    return WHARConfig(
        dataset_id="dummy",
        download_url="https://example.invalid",
        sampling_freq=50,
        num_of_subjects=1,
        num_of_activities=1,
        num_of_channels=1,
        datasets_dir=".",
        parse=lambda _raw_dir, _dataset_dir: (  # type: ignore[return-value]
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        ),
        activity_names=["a"],
        sensor_channels=["x"],
        window_time=1.0,
        window_overlap=0.0,
        seed=seed,
    )


class TestRandomSplitter(unittest.TestCase):
    def test_sizes_and_no_overlap(self) -> None:
        cfg = _dummy_cfg(seed=123)
        splitter = RandomSplitter(cfg, train_percentage=0.6, val_percentage=0.2, test_percentage=0.2)

        session_df = pd.DataFrame(
            {
                "session_id": [1],
                "subject_id": [1],
            }
        )
        window_df = pd.DataFrame({"session_id": [1] * 10}, index=list(range(10)))

        split = splitter.get_splits(session_df, window_df)[0]

        self.assertEqual(len(split.train_indices), 6)
        self.assertEqual(len(split.val_indices), 2)
        self.assertEqual(len(split.test_indices), 2)

        self.assertFalse(set(split.train_indices) & set(split.val_indices))
        self.assertFalse(set(split.train_indices) & set(split.test_indices))
        self.assertFalse(set(split.val_indices) & set(split.test_indices))

    def test_reproducible_with_seed(self) -> None:
        cfg = _dummy_cfg(seed=42)
        splitter_a = RandomSplitter(cfg, train_percentage=0.7, val_percentage=0.1, test_percentage=0.2)
        splitter_b = RandomSplitter(cfg, train_percentage=0.7, val_percentage=0.1, test_percentage=0.2)

        session_df = pd.DataFrame({"session_id": [1], "subject_id": [1]})
        window_df = pd.DataFrame({"session_id": [1] * 20}, index=list(range(20)))

        split_a = splitter_a.get_splits(session_df, window_df)[0]
        split_b = splitter_b.get_splits(session_df, window_df)[0]

        self.assertEqual(split_a.train_indices, split_b.train_indices)
        self.assertEqual(split_a.val_indices, split_b.val_indices)
        self.assertEqual(split_a.test_indices, split_b.test_indices)


if __name__ == "__main__":
    unittest.main()

