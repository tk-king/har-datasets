from typing import Callable, Tuple
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from har_datasets.parsing.loading import load_df
from har_datasets.preparing.normalizing import (
    min_max,
    normalize_globally,
    normalize_per_sample,
    normalize_per_subject,
    standardize,
)
from har_datasets.preparing.resampling import resample
from har_datasets.preparing.selecting import select_activities, select_channels
from har_datasets.preparing.windowing import generate_windows
from har_datasets.config.config import Config, SplitType, NormType


class HARDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, cfg: Config, parse: Callable[[str], pd.DataFrame]):
        super().__init__()

        self.cfg = cfg

        # load dataframe
        df = load_df(
            url=cfg.dataset.url,
            datasets_dir=cfg.dataset.dir,
            csv_file=cfg.dataset.csv_file,
            parse=parse,
        )

        # apply selections
        df = select_activities(df=df, activity_ids=cfg.dataset.selections.activity_ids)
        df = select_channels(df=df, channels=cfg.dataset.selections.channels)

        # apply resampling
        if cfg.common.resampling_freq is not None:
            df = resample(
                df=df,
                sampling_freq=cfg.dataset.sampling_freq,
                resampling_freq=cfg.common.resampling_freq,
            )

        # apply global or per subject normalization
        match cfg.common.normalization:
            case NormType.STD_GLOBALLY:
                df = normalize_globally(df, standardize)
            case NormType.MIN_MAX_GLOBALLY:
                df = normalize_globally(df, min_max)
            case NormType.STD_PER_SUBJECT:
                df = normalize_per_subject(df, standardize)
            case NormType.MIN_MAX_PER_SUBJECT:
                df = normalize_per_subject(df, min_max)

        # generate windows and window index
        self.window_index, self.windows = generate_windows(
            df=df,
            window_size=cfg.common.sliding_window.window_size,
            displacement=cfg.common.sliding_window.displacement,
        )

        # apply per sample normalization
        match cfg.common.normalization:
            case NormType.STD_PER_SAMPLE:
                df = normalize_per_sample(self.windows, standardize)
            case NormType.MIN_MAX_PER_SAMPLE:
                df = normalize_per_sample(self.windows, min_max)

        # specify split indices depending on split type
        match cfg.dataset.split.split_type:
            case SplitType.GIVEN:
                self.train_indices = self.window_index[
                    self.window_index["subject_id"].isin(
                        cfg.dataset.split.given_split.train_subject_ids
                    )
                ].index.to_list()

                self.test_indices = self.window_index[
                    self.window_index["subject_id"].isin(
                        cfg.dataset.split.given_split.test_subject_ids
                    )
                ].index.to_list()

                self.val_indices = self.window_index[
                    self.window_index["subject_id"].isin(
                        cfg.dataset.split.given_split.val_subject_ids
                    )
                ].index.to_list()

            case SplitType.SCV:
                test_subject_ids = cfg.dataset.split.scv_split.test_groups[
                    cfg.dataset.split.scv_split.test_group_index
                ]

                self.train_indices = self.window_index[
                    ~self.window_index["subject_id"].isin(test_subject_ids)
                ].index.to_list()

                self.test_indices = self.window_index[
                    self.window_index["subject_id"].isin(test_subject_ids)
                ].index.to_list()

                self.val_indices = []

    def get_dataloaders(
        self, train_batch_size: int, train_shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # specify split
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)

        batch_size = train_batch_size or self.cfg.dataset.training.batch_size

        # create dataloaders from split
        train_loader = DataLoader(train_set, batch_size, train_shuffle)
        test_loader = DataLoader(test_set, 1, False)
        val_loader = DataLoader(val_set, 1, False)

        return train_loader, test_loader, val_loader

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # get class label of window
        label = self.window_index.loc[index]["activity_id"]
        assert isinstance(label, np.integer)

        # get window as sample
        window_id = self.window_index.loc[index]["window_id"]
        assert isinstance(window_id, np.integer)
        window = self.windows[window_id]

        # drop index since not a feature
        window = window.reset_index(drop=True)

        x = torch.tensor(window.values, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.long)

        return x, y
