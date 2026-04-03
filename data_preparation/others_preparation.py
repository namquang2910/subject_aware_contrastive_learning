import wfdb
import pytz
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from datetime import datetime as dt
from utlis import ecg_preprocessing, create_segments, agg_labels
from config import DATASET_DEFAULTS, DOWNSAMPLE_SR
import os
import pickle

def load_subject_pickle_data(data_dir, sub_id, dataset):
    """Load one subject's ECG + labels from a pickle file."""
    if dataset == "wesad":
        sub_path = os.path.join(data_dir, sub_id, f"{sub_id}.pkl")
    else:
        sub_path = os.path.join(data_dir, sub_id)

    data_sr = DATASET_DEFAULTS[dataset]["data_sr"]
    sub_data = pickle.load(open(sub_path, "rb"), encoding="latin1")
    labels   = np.array(sub_data["label"])

    if dataset == "wesad":
        ecg_raw = np.array(sub_data["signal"]["chest"]["ECG"][:, 0])
    else:
        ecg_raw = np.array(sub_data["ECG"])

    ecg_signal = ecg_preprocessing(ecg_raw, sample_rate=data_sr,
                                    downsample_rate=DOWNSAMPLE_SR)

    start_dt   = dt(2017, 11, 28, 0, 0, 0, 0, tzinfo=pytz.UTC)
    label_freq = pd.DateOffset(seconds=1 / data_sr)
    ecg_freq   = pd.DateOffset(seconds=1 / DOWNSAMPLE_SR)

    label_times = pd.date_range(start=start_dt, periods=labels.shape[0],
                                freq=label_freq, tz="UTC")
    ecg_times   = pd.date_range(start=start_dt, periods=ecg_signal.shape[0],
                                freq=ecg_freq, tz="UTC")

    label_df = pd.DataFrame({"label_sample_timestamp_utc": label_times, "y": labels})
    ecg_df   = pd.DataFrame({"ecg_sample_timestamp_utc": ecg_times,
                              "ecg": ecg_signal.flatten()})

    ecg_df = pd.merge_asof(
        ecg_df.sort_values("ecg_sample_timestamp_utc"),
        label_df.sort_values("label_sample_timestamp_utc"),
        left_on="ecg_sample_timestamp_utc",
        right_on="label_sample_timestamp_utc",
        direction="nearest",
    )
    ecg_df.drop(columns="label_sample_timestamp_utc", inplace=True)
    ecg_df.set_index("ecg_sample_timestamp_utc", inplace=True, drop=False)
    return ecg_df

def process_subject_data(data_dir, sub_id, output_dir,
                          segment_length, segment_stride,dataset):
    ecg_df = load_subject_pickle_data(data_dir, sub_id, dataset)

    
    df_labelled, df_unlabelled = create_segments(
        ecg_df, segment_length, segment_stride, dataset, agg_labels
    )
    print(f"  Labelled segments: {df_labelled.shape[0]}")

    subject_out = os.path.join(output_dir, sub_id)
    os.makedirs(subject_out, exist_ok=True)
    df_labelled.to_parquet(
        os.path.join(subject_out, "ECG_labelled.parquet"), index=False
    )
    df_unlabelled.to_parquet(
        os.path.join(subject_out, "ECG_unlabelled.parquet"), index=False
    )


def load_all_subjects(data_dir, output_dir, segment_length, segment_stride, dataset):
    subjects = sorted(os.listdir(data_dir))
    for sub_id in tqdm(subjects):
        if sub_id.endswith(".pkl") or sub_id.startswith("S"):
            print("Processing subject:", sub_id)
            process_subject_data(
                data_dir, sub_id, output_dir,
                segment_length=segment_length,
                segment_stride=segment_stride,
                dataset=dataset,
            )