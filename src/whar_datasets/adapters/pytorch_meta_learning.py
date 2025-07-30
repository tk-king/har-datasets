import random
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_label, get_window
from whar_datasets.core.splitting import get_split
from whar_datasets.core.utils.normalization import normalize_windows
from whar_datasets.core.utils.loading import load_session_metadata, load_window_metadata
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class PytorchMetaLearningAdapter(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        super().__init__()

        self.cfg = cfg

        dirs = preprocess(cfg, override_cache)
        self.cache_dir, self.windows_dir, self.normalized_dir, self.hashes_dir = dirs

        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata = load_window_metadata(self.cache_dir)

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        self.seed = cfg.seed

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def get_dataloaders(
        self,
        context_size: int,
        train_batch_size: int,
        train_shuffle: bool = True,
        subj_cross_val_group_index: int | None = None,
        override_cache: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # get split indices from config
        self.train_indices, self.val_indices, self.test_indices = get_split(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            subj_cross_val_group_index,
        )

        # normalize windows
        self.window_metadata, self.windows = normalize_windows(
            self.cfg,
            self.train_indices,
            self.windows_dir,
            self.normalized_dir,
            self.hashes_dir,
            self.window_metadata,
            override_cache,
        )

        # Make sure all index sets are sets for fast lookup
        train_idx_set = set(self.train_indices)
        val_idx_set = set(self.val_indices)
        test_idx_set = set(self.test_indices)

        # STEP 1: Merge session and window metadata to get subject_id for each window
        merged_df = self.window_metadata.reset_index().merge(
            self.session_metadata, on="session_id"
        )  # index reset to preserve original row index

        # STEP 2: Group original indices by subject_id
        grouped_indices = merged_df.groupby("subject_id")["index"].apply(list)

        # STEP 3: Chunk and assign
        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i : i + size]

        self.contexts: List[List[int]] = []
        self.meta_train_indices = []
        self.meta_val_indices = []
        self.meta_test_indices = []

        for subject_indices in grouped_indices:
            for chunk in chunk_list(subject_indices, context_size):
                chunk_set = set(chunk)
                if chunk_set.issubset(train_idx_set):
                    self.meta_train_indices.append(len(self.contexts))
                    self.contexts.append(chunk)
                elif chunk_set.issubset(val_idx_set):
                    self.meta_val_indices.append(len(self.contexts))
                    self.contexts.append(chunk)
                elif chunk_set.issubset(test_idx_set):
                    self.meta_test_indices.append(len(self.contexts))
                    self.contexts.append(chunk)
                # skip mixed chunks

        # specify split subsets
        train_set = Subset(self, self.meta_train_indices)
        test_set = Subset(self, self.meta_val_indices)
        val_set = Subset(self, self.meta_test_indices)

        print(f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")

        # create dataloaders from split
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            generator=self.generator,
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=len(val_set),
            shuffle=False,
            generator=self.generator,
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            generator=self.generator,
        )

        return train_loader, val_loader, test_loader

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        indices = dataloader.dataset.indices  # type: ignore
        assert indices is not None

        return compute_class_weights(
            self.session_metadata, self.window_metadata.iloc[indices]
        )

    def __len__(self) -> int:
        return len(self.window_metadata)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # Get list of window indices for this chunk
        window_indices = self.contexts[index]

        x_list = []
        y_list = []

        for idx in window_indices:
            label = get_label(idx, self.window_metadata, self.session_metadata)
            window = get_window(
                idx, self.cfg, self.normalized_dir, self.window_metadata, self.windows
            )

            x = torch.tensor(window.values, dtype=torch.float32)
            y = torch.tensor(label, dtype=torch.long)

            x_list.append(x)
            y_list.append(y)

        # stack
        x_cat = torch.stack(x_list)
        y_cat = torch.stack(y_list)

        return y_cat, x_cat
