from collections import defaultdict
import os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.api import DataFrame as DataFrame
from har_datasets.old.dataparser import DataParser
from har_datasets.schema.schema import Config


class UCI_HAR_DataParser(DataParser):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.leave_one_out = True
        self.full_None_Overlapping = False
        self.Semi_None_Overlapping = True

        self.col_names = list(self.cfg.dataset.used_cols.values())
        # self.labelToId = {x: i for i, x in enumerate(cfg.dataset.label_map.keys())}
        self.ids_of_each_subj: Dict[np.integer, List[str]] = defaultdict(list)
        # TODO: why needed?

    def get_df_from_files(
        self,
        files: List[str],
        files_dir: str,
        subj_path: str,
        labels_path: str,
        slice_end: int,
    ) -> DataFrame:
        # create dict to store df for each channel, use file name without "train" as key
        dict = {
            file[:slice_end]: pd.read_csv(
                os.path.join(files_dir, file), header=None, sep="\\s+"
            )
            for file in files
        }  # (num_segs, seg_size)

        # the data is already segmented into 128 readings per window
        seg_size = 128
        for values in dict.values():
            assert values.shape[1] == seg_size

        # creates dfs from each dict, we ensure order of our specified cols through dict
        df = pd.DataFrame(
            np.stack(
                [dict[col].values.reshape(-1) for col in self.col_names],
                axis=1,
            ),
            columns=self.col_names,
        )  # (seg_size * num_segs, num_activities)

        # load subjects and labels as dfs
        subjects_df = pd.read_csv(subj_path, header=None, names=["subjects"])
        labels_df = pd.read_csv(labels_path, header=None, names=["labels"])
        # (num_segs, 1)

        # assert number of segs are the same
        assert df.shape[0] / seg_size == subjects_df.shape[0] == labels_df.shape[0]
        num_segs = labels_df.shape[0]

        subj_ids = []
        labels = []
        subjects = []

        # repeat the id and the label for each seg by the seg_size
        for i in range(num_segs):
            # get subject and label
            subj = subjects_df.loc[i, "subjects"]
            act = labels_df.loc[i, "labels"]
            assert isinstance(subj, np.integer) and isinstance(act, np.integer)

            # create and enter sub id
            subj_id = "{}_{}".format(subj, i)
            self.ids_of_each_subj[subj].append(subj_id)

            subj_ids.extend(seg_size * [subj_id])
            labels.extend(seg_size * [act])
            subjects.extend(seg_size * [subj])

        df["sub_id"] = subj_ids
        df["sub"] = subjects
        df["activity_id"] = labels
        # (seg_size * num_segs, num_activities + 3)

        return df

    def load_data(self, root_path: str) -> Tuple[DataFrame, Series]:
        print(" ----------------------- load all the data -------------------")

        # directories of raw data
        train_path = os.path.join(root_path, "train/Inertial Signals/")
        test_path = os.path.join(root_path, "test/Inertial Signals/")

        # file paths
        train_subj_path = os.path.join(root_path, "train/subject_train.txt")
        test_subj_path = os.path.join(root_path, "test/subject_test.txt")
        train_labels_path = os.path.join(root_path, "train/y_train.txt")
        test_labels_path = os.path.join(root_path, "test/y_test.txt")

        # get all files in train and test dirs
        train_files = os.listdir(train_path)
        test_files = os.listdir(test_path)

        # get train and test dfs
        train_df = self.get_df_from_files(
            files=train_files,
            files_dir=train_path,
            subj_path=train_subj_path,
            labels_path=train_labels_path,
            slice_end=-9,
        )  # (seg_size * num_segs, num_activities + 3)

        test_df = self.get_df_from_files(
            files=test_files,
            files_dir=test_path,
            subj_path=test_subj_path,
            labels_path=test_labels_path,
            slice_end=-8,
        )  # (seg_size * num_segs, num_activities + 3)

        # concat for varying splits
        df_all = pd.concat([train_df, test_df])
        # (seg_size * (num_segs_train + num_segs_test), num_activities + 3)

        # add another index based on sub_id
        df_dict: Dict[str, DataFrame] = {}
        for i in df_all.groupby("sub_id"):
            assert isinstance(i[0], str)
            df_dict[i[0]] = i[1]
            # (seg_size, num_activities + 3)

        df_all = pd.concat(df_dict)

        # label transformation
        # df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # set index
        df_all = df_all.set_index("sub_id")

        # split x and y
        data_x = df_all.iloc[:, :-1]
        data_y = df_all.iloc[:, -1]

        data_x = data_x.reset_index()

        return data_x, data_y
