import numpy as np
import neurokit2 as nk
import pandas as pd
from config import DATASET_DEFAULTS, DOWNSAMPLE_SR
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

def moving_average(signal, window_size=10):
    """Compute moving average with the specified window size."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    return np.convolve(signal, np.ones(window_size) / window_size, mode="same")

def bandpass_filter(sig, fs, lowcut=0.5, highcut=100.0, order=4):
    nyq = fs / 2.0
    high = min(highcut, nyq * 0.999)
    sos = butter(order, [lowcut / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, sig, axis=0)

def ecg_preprocessing(signal, sample_rate,
                       lowcut=0.5, highcut=100,
                       ma_window=10, downsample_rate=128):
    
    if signal.ndim == 2:
        # If the signal has multiple leads, apply the filter to each lead
        kernel = np.ones(ma_window) / ma_window
        band_passed = bandpass_filter(signal, fs=sample_rate, lowcut=lowcut, highcut=highcut)
        smoothed = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, band_passed)
    else:
        band_passed = nk.signal_filter(
            signal, sampling_rate=sample_rate,
            lowcut=lowcut, highcut=highcut,
            method="butterworth_zi", order=2,
        )
        smoothed    = moving_average(band_passed, window_size=ma_window)
    downsampled = nk.signal_resample(
        smoothed, sampling_rate=sample_rate,
        desired_sampling_rate=downsample_rate,
    )
    return downsampled
    

def load_config(dataset):
    """Load dataset-specific configuration parameters."""
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Dataset '{dataset}' not found in DATASET_DEFAULTS.")
    
    config = DATASET_DEFAULTS[dataset]
    segment_length = config.get("segment_length", None)
    segment_stride = config.get("segment_stride", None)
    data_sr        = config.get("data_sr", None)
    label_sets     = config.get("label_sets", None)
    
    return segment_length, segment_stride, data_sr, label_sets

def agg_labels(label_list, dataset):
    """Keep segment only if all samples share the same label within valid_labels."""
    valid_labels = DATASET_DEFAULTS[dataset].get("label_sets", None) or DATASET_DEFAULTS[dataset].get("labels_sets", None)
    
    label_set = set(label_list)
    if len(label_set) != 1:
        return np.nan
    l = list(label_set)[0]
    return l if l in valid_labels else np.nan


def create_segments(ecg_df, segment_length, segment_stride, dataset, agg_fn):
    ecg_segs, label_segs, left_buffers, right_buffers = [], [], [], []
    print("Label distribution:", np.unique(ecg_df["y"]))

    for i in range(1, len(ecg_df) - segment_length, segment_stride):
        seg   = ecg_df["ecg"][i : i + segment_length]
        label = agg_fn(ecg_df["y"][i : i + segment_length], dataset)

        ecg_segs.append(list(seg))
        label_segs.append(label)

        # left buffer
        if i >= segment_length:
            left_buffers.append(list(ecg_df["ecg"][i - segment_length : i]))
        else:
            buf = np.full_like(seg, np.nan)
            tail = ecg_df["ecg"][:i]
            buf[-tail.shape[0]:] = tail
            left_buffers.append(buf)

        # right buffer
        if i + 2 * segment_length < len(ecg_df):
            right_buffers.append(
                list(ecg_df["ecg"][i + segment_length : i + 2 * segment_length])
            )
        else:
            buf = np.full_like(seg, np.nan)
            tail = ecg_df["ecg"][i + segment_length:]
            buf[:tail.shape[0]] = tail
            right_buffers.append(buf)

    ecg_segs     = np.array(ecg_segs)
    left_buffers = np.array(left_buffers)
    right_buffers= np.array(right_buffers)
    label_segs   = np.array(label_segs)
    keep_mask    = ~np.isnan(label_segs)

    print("Segment label counts:", np.unique(label_segs, return_counts=True))

    df_labelled = pd.DataFrame({
        "x":              ecg_segs[keep_mask].tolist(),
        "x_left_buffer":  left_buffers[keep_mask].tolist(),
        "x_right_buffer": right_buffers[keep_mask].tolist(),
        "y":              label_segs[keep_mask].tolist(),
    })
    df_unlabelled = pd.DataFrame({
        "x":              ecg_segs.tolist(),
        "x_left_buffer":  left_buffers.tolist(),
        "x_right_buffer": right_buffers.tolist(),
        "y":              label_segs.tolist(),
    })
    return df_labelled, df_unlabelled

def create_segments_no_labels(ecg_array, segment_length, segment_stride):
    ecg_array = np.array(ecg_array)
    n_samples = ecg_array.shape[0]
    buf_shape = ecg_array.shape[1:] if ecg_array.ndim > 1 else ()  # (12,) or ()

    starts = list(range(1, n_samples - segment_length, segment_stride)) \
             if n_samples > segment_length else [0]
    ecg_segs = np.stack([ecg_array[i : i + segment_length] for i in starts])
    left_buffers, right_buffers = [], []

    def empty_buf():
        return np.full((segment_length, *buf_shape), np.nan, dtype=np.float32)

    for i in tqdm(starts, desc="  Buffering", leave=False):
        if i == 0:
            left_buffers.append(empty_buf())
        elif i >= segment_length:
            left_buffers.append(ecg_array[i - segment_length : i])
        else:
            buf = empty_buf()
            buf[-i:] = ecg_array[:i]
            left_buffers.append(buf)

        if i + 2 * segment_length < n_samples:
            right_buffers.append(ecg_array[i + segment_length : i + 2 * segment_length])
        else:
            buf = empty_buf()
            tail = ecg_array[i + segment_length:]
            buf[:len(tail)] = tail
            right_buffers.append(buf)

    return pd.DataFrame({
        "x":              list(ecg_segs),
        "x_left_buffer":  left_buffers,
        "x_right_buffer": right_buffers,
    })