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
        "data_sr":        512,
        "label_sets":      [0, 1], #0 for relax, 1 for ppt,
        "path_to_subject": "/home/s223149341/SSL-invariance-Subject_Project_model/data/VerBIO_v2/PRE/participant_id.csv"
    }
}

