import os

import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from utlis import ecg_preprocessing, create_segments_no_labels
from config import DATASET_DEFAULTS, DOWNSAMPLE_SR

# Lead order as stored in the PTB-XL WFDB records
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def read_ptbxl_record(rec_path, lead="II"):
    #Load the files 
    sig, _   = wfdb.rdsamp(rec_path)
    
    #load the specific leads
    if lead is None:
        leads = LEAD_NAMES
        x = sig.astype(np.float32)
        warnings.warn("No lead specified. All leads will be processed. This may take a long time.")
    else:
        lead_idx = LEAD_NAMES.index(lead)
        x        = sig[:, lead_idx].astype(np.float32)

    x_processed  = ecg_preprocessing(
        x, sample_rate=DATASET_DEFAULTS["ptbxl"]["data_sr"],
        downsample_rate=DOWNSAMPLE_SR,
    )
    return x_processed


def load_all_ptbxl(data_dir, output_dir, segment_length=None, segment_stride=None, lead="II"):
    ptbxl_df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    df_list  = []

    for ecg_id, row in tqdm(ptbxl_df.iterrows(), total=ptbxl_df.shape[0]):
        
        print("Processing record:", ecg_id)
        rec_path = os.path.join(data_dir, row["filename_hr"])  # 500 Hz records
        try:
            ecg_array = read_ptbxl_record(rec_path, lead=lead)
        except Exception as e:
            print(f"  Warning: could not read {rec_path} ({e}). Skipping.")
            continue

        print(f"  ECG length: {len(ecg_array)} samples")
        df_unlabelled = create_segments_no_labels(ecg_array, segment_length, segment_stride)
        df_unlabelled["subject_id"] = row["patient_id"]
        df_unlabelled["ecg_id"]     = ecg_id
        df_list.append(df_unlabelled)

    data_df  = pd.concat(df_list, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ptbxl_unlabelled.parquet")

    print(f"Total segments: {data_df.shape[0]}")
    print(f"Saving to: {os.path.abspath(out_path)}")
    np.savez_compressed(out_path,
    x=np.stack(data_df["x"].values),
    x_left=np.stack(data_df["x_left_buffer"].values),
    x_right=np.stack(data_df["x_right_buffer"].values),
    subject_id=data_df["subject_id"].values,
    ecg_id=data_df["ecg_id"].values)
    print("Done.")