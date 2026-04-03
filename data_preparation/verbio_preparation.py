
import wfdb
import pytz
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from datetime import datetime as dt
import glob
import os
from pathlib import Path
from utlis import ecg_preprocessing, create_segments, agg_labels
from config import DATASET_DEFAULTS, DOWNSAMPLE_SR

def load_subject_verbio_data(pid, data_dir= None):
    #Load the data from files
    def load_data(files, section = "PRE"):
        if len(files) == 0: 
            print(f"Cannot load {section} files for {pid=} as do not exist")
            return None 
        else:
            return [
                (f"{pid}_{section}_{filename.split('_')[-1].split('.')[0]}", 
                 pd.read_csv(filename)) for filename in files
            ]
    
    pre_condition_path = Path(f"{data_dir}/PRE/Actiwave/")
    post_condition_path = Path(f"{data_dir}/POST/Actiwave/")
    test_condition_path = Path(f"{data_dir}/TEST/Actiwave/")

    pre_ecg_files = [str(f) for f in pre_condition_path.rglob(f"{pid}/ECG_*.csv")] 
    post_ecg_files = [str(f) for f in post_condition_path.rglob(f"{pid}/ECG_*.csv")] 
    test_ecg_files = [str(f) for f in test_condition_path.rglob(f"{pid}/ECG_*.csv")]
    
    # Load the pre data
    pre_ecg_data = load_data(pre_ecg_files, section="PRE")
    post_ecg_data = load_data(post_ecg_files, section="POST")
    test_ecg_data = load_data(test_ecg_files, section="TEST")

    all_ecg_data = []
    if pre_ecg_data is not None:
        all_ecg_data.extend(pre_ecg_data)
    if post_ecg_data is not None:
        all_ecg_data.extend(post_ecg_data)
    if test_ecg_data is not None:
        all_ecg_data.extend(test_ecg_data)

    return all_ecg_data



def process_verbio_subject_data(data_dir, sub_id, output_dir,
                               segment_length, segment_stride):
    
    all_labelled, all_unlabelled = [], []
    ecg_df = load_subject_verbio_data(sub_id, data_dir)
    # Process the loaded data similarly to process_subject_data
    for ecg_info, ecg_data in ecg_df:
        #Skip the segment if it has fewer samples than segment_length
        if ecg_data.shape[0] < segment_length:
            print(f"Skipping {ecg_info} as it has fewer samples than segment_length")
            continue

        ecg_raw = ecg_data["ECG"].values
        ecg_signal = ecg_preprocessing(ecg_raw, sample_rate=DATASET_DEFAULTS["verbio"]["data_sr"],
                                        downsample_rate=DOWNSAMPLE_SR)
        labels = []
        # Assign labels based on the condition in the filename
        if "RELAX" in ecg_info:
            labels = [0] * ecg_signal.shape[0]
        elif "PPT" in ecg_info:
            labels = [1] * ecg_signal.shape[0]
        else:
            labels = [np.nan] * ecg_signal.shape[0]
        print(f"Processed {ecg_info} with {ecg_signal.shape[0]} samples and label {labels[0]}")
        ecg_data = pd.DataFrame({"ecg": ecg_signal, "y": labels})
        
        df_labelled, df_unlabelled = create_segments(    ecg_data, segment_length, segment_stride, "verbio", agg_labels)
        subject_out = os.path.join(output_dir, sub_id)
        all_labelled.append(df_labelled)
        all_unlabelled.append(df_unlabelled)

    subject_out = os.path.join(output_dir, sub_id)
    os.makedirs(subject_out, exist_ok=True)
    pd.concat(all_labelled, ignore_index=True).to_parquet(
        os.path.join(subject_out, "ECG_labelled.parquet"), index=False
    )
    pd.concat(all_unlabelled, ignore_index=True).to_parquet(
        os.path.join(subject_out, "ECG_unlabelled.parquet"), index=False
    )


def load_all_verbio_subjects(data_dir, output_dir, segment_length, segment_stride):
    subject_ids = pd.read_csv(DATASET_DEFAULTS["verbio"]["path_to_subject"])
    for ix, row in tqdm(subject_ids.iterrows()):
        sub_id = row["PID"]
        print("Processing subject:", sub_id)
        process_verbio_subject_data(
            data_dir, sub_id, output_dir,
            segment_length=segment_length,
            segment_stride=segment_stride,
        )