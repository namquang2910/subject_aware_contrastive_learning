"""
Dataset preparation script for ECG stress datasets.

Supported datasets: wesad, swell, physionet2017

Usage examples:
    python dataset_preparation.py --dataset wesad --data_dir /path/to/WESAD
    python dataset_preparation.py --dataset swell --data_dir /path/to/SWELL --segment_length 640
    python dataset_preparation.py --dataset physionet2017 --data_dir /path/to/physionet --output_dir /out
"""

import os
import glob
import argparse
from pathlib import Path
from pathlib import Path
import pickle

import wfdb
import pytz
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from datetime import datetime as dt
from others_preparation import load_all_subjects
from psy_preparation import load_all_physionet
from verbio_preparation import load_all_verbio_subjects

DOWNSAMPLE_SR  = 128

# Per-dataset defaults
DATASET_DEFAULTS = {
    "wesad": {
        "segment_length": 1280,
        "segment_stride": 64,
        "data_sr":        700,
        "label_sets":         [1, 2, 3], # 2 for stress and 1,3 for non_stress
    },
    "swell": {
        "segment_length": 1280,
        "segment_stride": 320,
        "data_sr":        2048,
        "label_sets":         [0, 2, 3], #0 for non-stress, 2,3 for stess
    },
    "physionet2017": {
        "segment_length": 1280,
        "segment_stride": 64,
        "data_sr":        300,
        "label_sets":      None,  # No labels for pretraining
    },
    "stressid": {
        "segment_length": 1280,
        "segment_stride": 64,
        "data_sr":        500,
        "label_sets":      [0, 1],
    },
    "verbio": {
        "segment_length": 1280,
        "segment_stride": 64,
        "data_sr":        None,
        "label_sets":      [0, 1], #0 for relax, 1 for ppt,
        "path_to_subject": "/home/s223149341/SSL-invariance-Subject_Project_model/data/VerBIO_v2/PRE/participant_id.csv"
    }
}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ECG datasets (WESAD / SWELL / PhysioNet 2017 / STRESSID) "
                    "into segmented parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset", required=True,
        choices=["wesad", "swell", "physionet2017", "stressid", "verbio"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the raw dataset directory.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write output parquet files. "
             "Defaults to <data_dir>/../<dataset>_prepared.",
    )
    parser.add_argument(
        "--segment_length", type=int, default=None,
        help="Number of samples per segment (post-downsampling). "
             "Dataset defaults: wesad=1280, swell=1280, physionet2017=1280.",
    )
    parser.add_argument(
        "--segment_stride", type=int, default=None,
        help="Stride between consecutive segments. "
             "Dataset defaults: wesad=64, swell=320, physionet2017=64.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]

    segment_length = args.segment_length if args.segment_length is not None \
                     else defaults["segment_length"]
    segment_stride = args.segment_stride if args.segment_stride is not None \
                     else defaults["segment_stride"]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.data_dir)),
            f"{args.dataset}_prepared",
        )

    print(f"Dataset       : {args.dataset}")
    print(f"Data dir      : {args.data_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Segment length: {segment_length}")
    print(f"Segment stride: {segment_stride}")
    print()

    if args.dataset in ["wesad", "swell", "stressid"]:
        load_all_subjects(args.data_dir, output_dir, segment_length, segment_stride, args.dataset)

    elif args.dataset == "physionet2017":
        load_all_physionet(args.data_dir, output_dir,
                          segment_length, segment_stride)

    elif args.dataset == "verbio":
        load_all_verbio_subjects(args.data_dir, output_dir, segment_length, segment_stride)

if __name__ == "__main__":
    main()