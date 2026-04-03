import numpy as np
import neurokit2 as nk
import pandas as pd
from config import DATASET_DEFAULTS, DOWNSAMPLE_SR

def moving_average(signal, window_size=10):
    """Compute moving average with the specified window size."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    return np.convolve(signal, np.ones(window_size) / window_size, mode="same")


def ecg_preprocessing(signal, sample_rate,
                       lowcut=0.5, highcut=100,
                       ma_window=10, downsample_rate=128):
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