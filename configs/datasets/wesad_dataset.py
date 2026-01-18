"""
Class for dataset that consists of EDA data from the WESAD dataset
"""
import os
import numpy as np
import pandas as pd
from datasets.dataset import DatasetWrapper


class WESADDataset(DatasetWrapper):
    def __init__(self, dataset_path, include_labels, split=None, split_file=None,
                 sub_sample_frac=None, sub_sample_count=None, sample_count_file=None,
                 data_views=None, transform_dict=None, transform_global=None):
        """
        :param dataset_path: path to folder containing sub-directories with subject data
        :param include_labels: if True, use labelled data; if false, use unlabelled segments
        :param split: split of data to use (should be key in split_file). If not provided, will use all data.
        :param split_file: path to file that contains information about which split each entry in the dataset is in.
        :param sub_sample_frac: If provided will randomly sub sample data to include only 'sub_sample_frac' frac of exs
        :param sample_count_file: if provided, look at this file for data counts
        :param data_views: If provided, will create multiple views of the data.
                           Dict storing parameters associated with the views to create.
        :param transform_dict: List of names of transforms to apply to signals if working with the transformed view
        :param transform_global: arguments to use when applying data transforms
        """
        self.include_labels = include_labels
        y_dim = 1 if self.include_labels else None
        y_var_type = "binary" if self.include_labels else None
        self.timezone = "utc"
        super().__init__(dataset_path, y_dim, y_var_type, split, split_file, sub_sample_frac, sub_sample_count,
                         sample_count_file, data_views, transform_dict, transform_global)

    def read_data(self):
        data_df = self.get_data_df()
        if self.split is not None:
            data_df = self.filter_by_split(data_df)

        # map y=2 to 1 and y=1 to 0
        def trf_labels(x):
            if x == 2:
                return 1
            return 0
        if "y" in data_df.columns:
            data_df["y"] = data_df["y"].apply(lambda x: trf_labels(x))
        # convert x entries to numpy arrays
        for col in ['x', 'x_left_buffer', 'x_right_buffer']:
            if col in data_df.columns:
                data_df[col] = data_df[col].apply(lambda x: np.array(x))
        data_df["subject_id_int"] = data_df["subject_id"].apply(lambda x: int(x[1:]))
        return data_df

    def read_subject_data(self, subject_dir):
        if self.include_labels:
            df = pd.read_parquet(os.path.join(subject_dir, "ECG_labelled.parquet"))
        else:
            df = pd.read_parquet(os.path.join(subject_dir, "ECG_unlabelled.parquet"))
        return df

    def _sub_dir_to_sub_id(self, sub_dir):
        return sub_dir
