import os
import random
import pickle
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from har_datasets.schema.schema import Config, ExpMode, ModelType
from har_datasets.old.normalizer import Normalizer
import pywt  # type: ignore
from abc import ABC, abstractmethod
from sklearn.utils import class_weight  # type: ignore


class DataParser(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.data_x: pd.DataFrame
        self.data_y: pd.Series
        self.slidingwindows: List[Tuple[str, int, int]]
        self.act_weights: List[float]
        self.freq_file_name: List[str]
        self.normalized_data_x: pd.DataFrame

        self.keep_activities = [
            activity_id
            for activity_id, (_, keep) in self.cfg.dataset.label_map.items()
            if keep
        ]

    @abstractmethod
    def load_data(self, root_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError

    def normalize(self, train_vali: pd.DataFrame) -> pd.DataFrame:
        train_vali_sensors = train_vali.iloc[:, 1:-1]
        normalizer = Normalizer(self.cfg.common.datanorm_type)
        normalizer.fit(train_vali_sensors)
        train_vali_sensors = normalizer.normalize(train_vali_sensors)
        train_vali = pd.concat(
            [train_vali.iloc[:, 0], train_vali_sensors, train_vali.iloc[:, -1]], axis=1
        )

        return train_vali

    def normalize_both(
        self, train_vali: pd.DataFrame, test: pd.DataFrame | None = None
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        train_vali_sensors = train_vali.iloc[:, 1:-1]
        normalizer = Normalizer(self.cfg.common.datanorm_type)
        normalizer.fit(train_vali_sensors)
        train_vali_sensors = normalizer.normalize(train_vali_sensors)
        train_vali = pd.concat(
            [train_vali.iloc[:, 0], train_vali_sensors, train_vali.iloc[:, -1]], axis=1
        )

        if test is None:
            return train_vali
        else:
            test_sensors = test.iloc[:, 1:-1]
            test_sensors = normalizer.normalize(test_sensors)
            test = pd.concat([test.iloc[:, 0], test_sensors, test.iloc[:, -1]], axis=1)
            return train_vali, test

    def apply_differencing(self, df: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = df.columns[:-1]
        diff_cols = ["diff_" + col for col in sensor_cols]

        diff_data = df[sensor_cols].groupby(df.index).diff()
        diff_data.rename(columns=dict(zip(sensor_cols, diff_cols)), inplace=True)
        diff_data.bfill(inplace=True)

        data = pd.concat([df.iloc[:, :-1], diff_data, df.iloc[:, -1]], axis=1)

        return data.reset_index()

    def get_sliding_window_index(
        self, data_x: pd.DataFrame, data_y_s: pd.Series
    ) -> List[Tuple[str, int, int]]:
        print("----------------------- Get the Sliding Window -------------------")

        data_y = data_y_s.reset_index()
        data_x["activity_id"] = data_y["activity_id"]

        # Creates a unique act_block identifier for each continuous segment of data with the same activity_id and sub_id.
        # Essentially identifies where activity or subject changes and assigns a new block number
        data_x["act_block"] = (
            (
                (data_x["activity_id"].shift(1) != data_x["activity_id"])
                | (data_x["sub_id"].shift(1) != data_x["sub_id"])
            )
            .astype(int)
            .cumsum()
        )

        windowsize = self.cfg.common.sliding_window.windowsize
        displacement = self.cfg.common.sliding_window.displacement

        window_index: List[Tuple[str, int, int]] = []
        for index in tqdm(data_x.act_block.unique()):
            # df per act_block
            temp_df = data_x[data_x["act_block"] == index]

            # only one activity per act_block
            assert temp_df["activity_id"].nunique() == 1

            if temp_df["activity_id"].unique()[0] in self.keep_activities:
                assert len(temp_df["sub_id"].unique()) == 1

                sub_id = str(temp_df["sub_id"].unique()[0])
                start = int(temp_df.index[0])
                end = int(start + windowsize)

                while end <= temp_df.index[-1] + 1:
                    window_index.append((sub_id, start, end))

                    start += displacement
                    end = start + windowsize

        return window_index

    def get_class_weights(self) -> List[float]:
        # class_transform = {x: i for i, x in enumerate(self.keep_activities)}

        y_of_all_windows = np.array(
            [
                int(self.data_y.iloc[window[1] : window[2]].mode().loc[0])
                for window in self.slidingwindows
            ]
        )

        act_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.array(self.keep_activities),
            y=y_of_all_windows,
        )

        return list(act_weights.round(4))

    def generate_spectrogram(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)

        freq_path = os.path.join(
            save_path,
            f"diff_{self.cfg.common.difference}_window_{self.cfg.common.sliding_window.windowsize}_step_{self.cfg.common.sliding_window.displacement}",
        )

        if os.path.exists(freq_path):
            print("----------------------- file are generated -----------------------")
            with open(os.path.join(freq_path, "freq_file_name.pickle"), "rb") as f:
                self.freq_file_name = pickle.load(f)
            return

        print("----------------------- spectrogram generating -----------------------")
        os.makedirs(freq_path, exist_ok=True)

        totalscal = self.cfg.common.sliding_window.sampling_freq + 1
        fc = pywt.central_frequency(self.cfg.common.wavename)
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)

        self.freq_file_name = []
        temp_data = self.normalize(self.data_x.copy())

        for window in tqdm(self.slidingwindows):
            sub_id, start_index, end_index = window
            name = f"{sub_id}_{start_index}_{end_index}"
            self.freq_file_name.append(name)

            sample_x = temp_data.iloc[start_index:end_index, 1:-1].values
            scalogram_list: List[np.ndarray] = []

            for j in range(sample_x.shape[1]):
                cwtmatr, _ = pywt.cwt(
                    sample_x[:, j],
                    scales,
                    self.cfg.common.wavename,
                    sampling_period=1.0 / self.cfg.common.sliding_window.sampling_freq,
                )
                scalogram_list.append(cwtmatr)

            scalogram = np.stack(scalogram_list)

            with open(os.path.join(freq_path, f"{name}.pickle"), "wb") as f:
                pickle.dump(scalogram, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(freq_path, "freq_file_name.pickle"), "wb") as f:
            pickle.dump(self.freq_file_name, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.freq_path = freq_path

    def load_and_prepare_data(self) -> None:
        self.data_x, self.data_y = self.load_data(self.cfg.dataset.root_path)

        if self.cfg.common.difference:
            self.data_x = self.apply_differencing(
                self.data_x.set_index("sub_id").copy()
            )

        self.slidingwindows = self.get_sliding_window_index(
            self.data_x.copy(), self.data_y.copy()
        )

        self.act_weights = self.get_class_weights()

        if self.cfg.common.model_type in [ModelType.freq, ModelType.cross]:
            self.generate_spectrogram(self.cfg.dataset.freq_save_path)

        # Handle experiment mode specifics
        match self.cfg.dataset.exp_mode:
            case ExpMode.Given | ExpMode.LOCV:
                self.num_of_cv = 5
                self.index_of_cv = 0
                self.step = int(len(self.slidingwindows) / self.num_of_cv)
                self.window_index_list = list(np.arange(len(self.slidingwindows)))
                random.shuffle(self.window_index_list)

                self.normalized_data_x = (
                    self.normalize(self.data_x.copy())
                    if self.cfg.common.datanorm_type is not None
                    else self.data_x.copy()
                )

            case ExpMode.LOCV:
                self.num_of_cv = len(self.cfg.dataset.keys.LOCV)
                self.index_of_cv = 0

            case _:
                self.num_of_cv = 1

    # def update_train_val_test_keys(self):
    #     # Access everything from self.cfg directly, no assignments except internal states
    #     exp_mode = self.cfg.dataset.exp_mode
    #     split_tag = self.cfg.dataset.split_tag
    #     train_vali_quote = self.cfg.common.train_vali_quote

    #     match exp_mode:
    #         case ExpMode.Given | ExpMode.LOCV:
    #             if exp_mode == ExpMode.LOCV:
    #                 print(
    #                     f"Leave one Out Experiment : The {self.index_of_cv} Part as the test"
    #                 )
    #                 test_keys = self.cfg.dataset.keys.LOCV[self.index_of_cv]
    #                 train_keys = [
    #                     k for k in self.cfg.dataset.keys.all if k not in test_keys
    #                 ]
    #                 self.index_of_cv += 1
    #             else:
    #                 train_keys = self.cfg.dataset.keys.train
    #                 test_keys = self.cfg.dataset.keys.test

    #             if self.cfg.common.datanorm_type is not None:
    #                 train_vali_x = pd.concat(
    #                     [
    #                         self.data_x[self.data_x[split_tag] == sub]
    #                         for sub in train_keys
    #                     ]
    #                 )
    #                 test_x = pd.concat(
    #                     [
    #                         self.data_x[self.data_x[split_tag] == sub]
    #                         for sub in test_keys
    #                     ]
    #                 )
    #                 train_vali_x, test_x = self.normalization(train_vali_x, test_x)
    #                 self.normalized_data_x = pd.concat(
    #                     [train_vali_x, test_x]
    #                 ).sort_index()
    #             else:
    #                 self.normalized_data_x = self.data_x.copy()

    #             train_vali_window_index = []
    #             test_window_index = []

    #             if split_tag == "sub":
    #                 all_test_keys = []
    #                 for sub in test_keys:
    #                     all_test_keys.extend(self.sub_ids_of_each_sub[sub])
    #             else:
    #                 all_test_keys = test_keys.copy()

    #             for idx, window in enumerate(self.slidingwindows):
    #                 sub_id = window[0]
    #                 if sub_id in all_test_keys:
    #                     test_window_index.append(idx)
    #                 else:
    #                     train_vali_window_index.append(idx)

    #             random.shuffle(train_vali_window_index)
    #             split_idx = int(train_vali_quote * len(train_vali_window_index))
    #             self.train_window_index = train_vali_window_index[:split_idx]
    #             self.vali_window_index = train_vali_window_index[split_idx:]
    #             self.test_window_index = test_window_index

    #         case ExpMode.SOCV | ExpMode.FOCV:
    #             print(
    #                 f"Overlapping random Experiment : The {self.index_of_cv} Part as the test"
    #             )

    #             start = self.index_of_cv * self.step
    #             end = (
    #                 (self.index_of_cv + 1) * self.step
    #                 if self.index_of_cv < self.num_of_cv - 1
    #                 else len(self.slidingwindows)
    #             )

    #             train_vali_index = (
    #                 self.window_index_list[:start] + self.window_index_list[end:]
    #             )
    #             self.test_window_index = self.window_index_list[start:end]

    #             split_idx = int(train_vali_quote * len(train_vali_index))
    #             self.train_window_index = train_vali_index[:split_idx]
    #             self.vali_window_index = train_vali_index[split_idx:]

    #             self.index_of_cv += 1

    #         case _:
    #             raise NotImplementedError

    #     class_transform = {
    #         x: i for i, x in enumerate(self.cfg.dataset.label_map.keys())
    #     }

    #     y_of_all_windows = [
    #         class_transform[self.data_y.iloc[window[1] : window[2]].mode().loc[0]]
    #         for window in (self.slidingwindows[i] for i in self.train_window_index)
    #     ]

    #     act_weights = class_weight.compute_class_weight(
    #         "balanced", range(len(class_transform)), y_of_all_windows
    #     )
    #     self.act_weights = act_weights.round(4)
    #     print("The class weights are:", self.act_weights)
