from typing import Callable, Tuple
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from har_datasets.parsing.loading import load_df
from har_datasets.preparing.selecting import (
    select_activities,
    select_channels,
    select_subjects,
)
from har_datasets.preparing.windowing import generate_windows
from har_datasets.config.schema import Config, ExpMode


class HARDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, cfg: Config, parse: Callable[[str], pd.DataFrame]):
        super().__init__()
        # load dataframe
        df = load_df(
            url=cfg.dataset.url,
            datasets_dir=cfg.dataset.dir,
            csv_file=cfg.dataset.file_name,
            parse=parse,
        )

        # apply selections
        df = select_subjects(df=df, subj_ids=cfg.dataset.subj_ids)
        df = select_activities(df=df, activity_ids=cfg.dataset.activity_ids)
        df = select_channels(df=df, channels=cfg.dataset.channels)

        # generate windows and window index
        self.window_index, self.windows = generate_windows(
            df=df,
            window_size=cfg.common.sliding_window.windowsize,
            displacement=cfg.common.sliding_window.displacement,
        )

        # specify split indices depending on experiment mode
        match cfg.dataset.exp_mode:
            case ExpMode.Given:
                self.train_indices = self.window_index[
                    self.window_index["subj_id"].isin(cfg.dataset.splits.train_subj_ids)
                ].index.to_list()

                self.test_indices = self.window_index[
                    self.window_index["subj_id"].isin(cfg.dataset.splits.test_subj_ids)
                ].index.to_list()

                self.val_indices = self.window_index[
                    self.window_index["subj_id"].isin(cfg.dataset.splits.val_subj_ids)
                ].index.to_list()
            case ExpMode.LSOCV:
                subj_ids_left_out = cfg.dataset.splits.LSOCV_subj_ids[
                    cfg.dataset.splits.LSOCV_index
                ]

                self.train_indices = self.window_index[
                    ~self.window_index["subj_id"].isin(subj_ids_left_out)
                ].index.to_list()

                self.test_indices = self.window_index[
                    self.window_index["subj_id"].isin(subj_ids_left_out)
                ].index.to_list()

                self.val_indices = []

    def get_dataloaders(
        self,
        batch_sizes: Tuple[int, int, int] | None = None,
        shuffles: Tuple[bool, bool, bool] = (True, False, False),
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # specify split (doesnt copy, provides only view)
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)

        batch_sizes = (
            self.cfg.common.batch_sizes if batch_sizes is None else batch_sizes
        )

        # create dataloaders from split
        train_loader = DataLoader(train_set, batch_sizes[0], shuffles[0])
        test_loader = DataLoader(test_set, batch_sizes[1], shuffles[1])
        val_loader = DataLoader(val_set, batch_sizes[2], shuffles[2])

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
