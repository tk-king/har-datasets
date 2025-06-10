import os
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from pandas import read_csv
from torch import Tensor
import torch
from torch.utils.data import Dataset
import mlcroissant as mlc  # type: ignore

from har_datasets.preparing.windowing import generate_windows
from har_datasets.schema.schema import Config


class DatasetHAR(Dataset):
    def __init__(self, cfg: Config, parse: Callable[[str], pd.DataFrame]):
        super().__init__()

        path = os.path.join(cfg.dataset.dir, cfg.dataset.file_name)

        # if file exists, load it, else parse from dataset and save
        if os.path.exists(path):
            df = read_csv(path)
        else:
            df = parse(cfg.dataset.dir)
            df.to_csv(path, index=True)

        # generate windows and window index
        self.window_index, self.windows = generate_windows(
            df=df,
            window_size=cfg.common.sliding_window.windowsize,
            displacement=cfg.common.sliding_window.displacement,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        label = self.window_index.loc[index]["activity_id"]
        assert isinstance(label, np.integer)

        window_id = self.window_index.loc[index]["window_id"]
        assert isinstance(window_id, np.integer)
        window = self.windows[window_id]

        # drop index since not a feature
        window = window.reset_index(drop=True)

        x = torch.tensor(window.values, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.long)

        return x, y


class CroissantDatasetHAR(Dataset):
    def __init__(
        self, cfg: Config, parse: Callable[[str], pd.DataFrame], ds: mlc.Dataset
    ):
        super().__init__()

        path = ds.metadata.to_json()["distribution"][0]["contentUrl"]
        assert isinstance(path, str)

        # if file exists, load it, else parse from dataset and save
        if os.path.exists(path):
            df = read_csv(path)
        else:
            df = parse(cfg.dataset.dir)
            df.to_csv(path, index=True)

        # generate windows and window index
        self.window_index, self.windows = generate_windows(
            df=df,
            window_size=cfg.common.sliding_window.windowsize,
            displacement=cfg.common.sliding_window.displacement,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        label = self.window_index.loc[index]["activity_id"]
        assert isinstance(label, np.integer)

        window_id = self.window_index.loc[index]["window_id"]
        assert isinstance(window_id, np.integer)
        window = self.windows[window_id]

        # drop index since not a feature
        window = window.reset_index(drop=True)

        x = torch.tensor(window.values, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.long)

        return x, y
