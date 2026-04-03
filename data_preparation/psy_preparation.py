import wfdb
import pytz
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from datetime import datetime as dt
import glob
import os
import pickle
from utlis import ecg_preprocessing, create_segments
from config import DATASET_DEFAULTS, DOWNSAMPLE_SR

def read_physionet_record(hea_path):
    rec_id   = os.path.splitext(os.path.basename(hea_path))[0]
    rec_path = os.path.join(os.path.dirname(hea_path), rec_id)
    sig, _   = wfdb.rdsamp(rec_path)
    x        = sig.squeeze().astype(np.float32)
    x        = ecg_preprocessing(x, sample_rate=DATASET_DEFAULTS["physionet2017"]["data_sr"],
                                  downsample_rate=DOWNSAMPLE_SR)
    return x


def create_segments_no_labels(ecg_array, segment_length, segment_stride):
    ecg_array  = np.array(ecg_array)
    starts     = list(range(1, len(ecg_array) - segment_length, segment_stride))
    ecg_segs   = np.stack([ecg_array[i : i + segment_length] for i in starts])
    left_buffers, right_buffers = [], []

    for i in tqdm(starts, desc="  Buffering", leave=False):
        if i >= segment_length:
            left_buffers.append(ecg_array[i - segment_length : i])
        else:
            buf = np.full(segment_length, np.nan, dtype=np.float32)
            buf[-i:] = ecg_array[:i]
            left_buffers.append(buf)

        if i + 2 * segment_length < len(ecg_array):
            right_buffers.append(ecg_array[i + segment_length : i + 2 * segment_length])
        else:
            buf = np.full(segment_length, np.nan, dtype=np.float32)
            tail = ecg_array[i + segment_length:]
            buf[:len(tail)] = tail
            right_buffers.append(buf)

    return pd.DataFrame({
        "x":              list(ecg_segs),
        "x_left_buffer":  left_buffers,
        "x_right_buffer": right_buffers,
    })


def load_all_physionet(data_dir, output_dir, segment_length, segment_stride):
    hea_paths = sorted(glob.glob(os.path.join(data_dir, "A*.hea")))
    subjects  = sorted({fname.split(".")[0] for fname in os.listdir(data_dir)})
    df_list   = []

    for hea, sub_id in tqdm(zip(hea_paths, subjects), total=len(subjects)):
        print("Processing record:", sub_id)
        ecg_array    = read_physionet_record(hea)
        df_unlabelled = create_segments_no_labels(ecg_array, segment_length, segment_stride)
        df_unlabelled["subject_id"] = sub_id
        df_list.append(df_unlabelled)

    data_df  = pd.concat(df_list, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "physionet2017_unlabelled.parquet")

    print(f"Total segments: {data_df.shape[0]}")
    print(f"Saving to: {os.path.abspath(out_path)}")
    data_df.to_parquet(out_path, index=False)
    print("Done.")