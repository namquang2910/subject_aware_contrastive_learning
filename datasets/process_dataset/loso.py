import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = (
    "/home/s223149341/"
    "SSL-invariance-Subject_Project_model/"
    "data/MDD/MDD_1280_64"
)

OUT_DIR = "MDD_L5SO"

N_FOLDS = 5

SHUFFLE = False
SEED = 42

# Validation size relative to the training+validation subjects
VAL_RATIO = 0.20


# ============================================================
# GET SUBJECT IDS
# ============================================================

def get_subject_ids(dataset_path):

    subject_ids = []

    for name in os.listdir(dataset_path):

        path = os.path.join(dataset_path, name)

        if not os.path.isdir(path):
            continue

        if name.startswith("."):
            continue

        subject_ids.append(str(name))

    # Keep 01, 02, ..., 10 in the correct order
    try:
        subject_ids = sorted(
            subject_ids,
            key=lambda x: int(x)
        )
    except ValueError:
        subject_ids = sorted(subject_ids)

    return subject_ids


# ============================================================
# GET SUBJECT LABEL
# ============================================================

def get_subject_label(subject_id):

    subject_path = os.path.join(
        DATASET_PATH,
        str(subject_id)
    )

    parquet_files = [
        f for f in os.listdir(subject_path)
        if f.endswith(".parquet")
    ]

    if len(parquet_files) == 0:
        raise ValueError(
            f"No parquet files found for subject {subject_id}"
        )

    labels = set()

    for file_name in parquet_files:

        file_path = os.path.join(
            subject_path,
            file_name
        )

        df = pd.read_parquet(file_path)

        if "y" not in df.columns:
            raise ValueError(
                f"Column 'y' not found in {file_path}"
            )

        labels.update(
            df["y"]
            .dropna()
            .unique()
            .tolist()
        )

    if len(labels) == 0:
        raise ValueError(
            f"Subject {subject_id} has no valid labels."
        )

    if len(labels) > 1:
        raise ValueError(
            f"Subject {subject_id} has multiple labels: "
            f"{sorted(labels)}"
        )

    return int(next(iter(labels)))


# ============================================================
# BUILD SUBJECT LABEL DATAFRAME
# ============================================================

def build_subject_label_dataframe(subject_ids):

    rows = []

    for subject_id in subject_ids:

        label = get_subject_label(subject_id)

        rows.append({
            "subject_id": str(subject_id),
            "label": int(label),
        })

    return pd.DataFrame(rows)


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(df, name):

    print(f"\n{name}")
    print("-" * 50)

    if len(df) == 0:
        print("EMPTY")
        return

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for label in counts.index:

        print(
            f"label {label}: "
            f"{counts[label]} subjects "
            f"({percentages[label]:.2f}%)"
        )

    print(f"total: {len(df)}")


# ============================================================
# CREATE STRATIFIED VALIDATION SET
# ============================================================

def create_validation_split(
    train_val_df,
    val_ratio=0.20,
    shuffle=False,
    seed=42,
):
    """
    Create validation set while preserving label ratio.

    Validation is created ONLY from train_val subjects.
    """

    if val_ratio <= 0:
        return train_val_df.copy(), pd.DataFrame(
            columns=train_val_df.columns
        )

    n_val = max(
        1,
        round(len(train_val_df) * val_ratio)
    )

    # Number of validation subjects per class
    class_counts = (
        train_val_df["label"]
        .value_counts()
        .sort_index()
    )

    n_classes = len(class_counts)

    if n_val < n_classes:
        raise ValueError(
            f"Not enough subjects for validation. "
            f"n_val={n_val}, classes={n_classes}"
        )

    # --------------------------------------------------------
    # Allocate validation subjects proportionally
    # --------------------------------------------------------

    allocation = {}

    for label, count in class_counts.items():

        allocation[label] = (
            count / len(train_val_df)
        ) * n_val

    # Floor first
    val_counts = {
        label: int(count)
        for label, count in allocation.items()
    }

    # Remaining slots
    remaining = (
        n_val
        - sum(val_counts.values())
    )

    # Give remaining slots to classes with
    # largest fractional parts
    fractional = sorted(
        allocation.keys(),
        key=lambda label:
            allocation[label] - val_counts[label],
        reverse=True,
    )

    for label in fractional[:remaining]:
        val_counts[label] += 1

    # --------------------------------------------------------
    # Select subjects
    # --------------------------------------------------------

    val_indices = []

    for label, n_samples in val_counts.items():

        group = train_val_df[
            train_val_df["label"] == label
        ]

        indices = list(group.index)

        if shuffle:
            import random

            rng = random.Random(seed)
            rng.shuffle(indices)

        selected = indices[:n_samples]

        val_indices.extend(selected)

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    val_df = train_val_df.loc[
        val_indices
    ].copy()

    train_df = train_val_df.drop(
        val_indices
    ).copy()

    return train_df, val_df


# ============================================================
# CREATE TEST FOLDS
# ============================================================

def create_test_folds(
    subject_df,
    n_folds=5,
    shuffle=False,
    seed=42,
):
    """
    Stratified subject-wise K-fold.

    Every subject appears in TEST exactly once.
    """

    subjects = subject_df["subject_id"].values
    labels = subject_df["label"].values

    class_counts = (
        subject_df["label"]
        .value_counts()
    )

    min_class_count = class_counts.min()

    if min_class_count < n_folds:

        raise ValueError(
            "\nCannot create StratifiedKFold.\n"
            f"n_folds = {n_folds}\n"
            f"Smallest class = {min_class_count} subjects\n"
            "\n"
            "Each class must contain at least "
            "N_FOLDS subjects."
        )

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )

    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(
        skf.split(subjects, labels)
    ):

        train_val_subjects = [
            subjects[i]
            for i in train_idx
        ]

        test_subjects = [
            subjects[i]
            for i in test_idx
        ]

        folds.append({
            "fold": fold_id,
            "train_val": train_val_subjects,
            "test": test_subjects,
        })

    return folds


# ============================================================
# GENERATE SPLITS
# ============================================================

def generate_splits(
    subject_df,
    folds=5,
    out_dir="MDD_L5SO",
    shuffle=False,
    seed=42,
):
    """
    Generate subject-wise folds where:

        TRAIN = remaining subjects
        VAL   = test subjects
        TEST  = same subjects as VAL

    Therefore:

        VAL subjects == TEST subjects
        VAL samples  == TEST samples

    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Overall distribution
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FULL DATASET")
    print("=" * 70)

    print_distribution(
        subject_df,
        "ALL SUBJECTS"
    )

    # --------------------------------------------------------
    # Stratified folds
    # --------------------------------------------------------

    fold_data = create_test_folds(
        subject_df,
        n_folds=folds,
        shuffle=shuffle,
        seed=seed,
    )

    # --------------------------------------------------------
    # Each fold
    # --------------------------------------------------------

    for fold_info in fold_data:

        fold_id = fold_info["fold"]

        # ----------------------------------------------------
        # TEST subjects
        # ----------------------------------------------------

        test_subjects = fold_info["test"]

        # ----------------------------------------------------
        # TRAIN subjects
        # ----------------------------------------------------

        train_subjects = fold_info["train_val"]

        # ----------------------------------------------------
        # Build TEST dataframe
        # ----------------------------------------------------

        test_df = subject_df[
            subject_df["subject_id"].isin(
                test_subjects
            )
        ].copy()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Validation is EXACTLY the same as TEST
        #
        # We use .copy() so they are separate dataframes
        # but contain exactly the same subjects/samples.
        # ----------------------------------------------------

        val_df = test_df.copy()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_df = subject_df[
            subject_df["subject_id"].isin(
                train_subjects
            )
        ].copy()

        # ----------------------------------------------------
        # Add split
        # ----------------------------------------------------

        train_df["split"] = "train"

        val_df["split"] = "val"

        test_df["split"] = "test"

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        fold_df = pd.concat(
            [
                train_df,
                val_df,
                test_df,
            ],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        out_path = os.path.join(
            out_dir,
            f"fold_{fold_id:02d}.csv"
        )
        fold_df.drop(columns=["label"], inplace=True)
        fold_df.to_csv(
            out_path,
            index=False
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold_id:02d}")
        print("=" * 70)

        print_distribution(
            train_df,
            "TRAIN"
        )

        print_distribution(
            val_df,
            "VALIDATION"
        )

        print_distribution(
            test_df,
            "TEST"
        )

        # ----------------------------------------------------
        # Check VAL == TEST
        # ----------------------------------------------------

        val_subjects_sorted = sorted(
            val_df["subject_id"].tolist()
        )

        test_subjects_sorted = sorted(
            test_df["subject_id"].tolist()
        )

        assert (
            val_subjects_sorted
            == test_subjects_sorted
        ), (
            f"VAL and TEST subjects are different "
            f"in fold {fold_id}"
        )

        print(
            "\nVAL == TEST subjects: TRUE"
        )

        print("\nTRAIN:")
        print(
            train_df["subject_id"].tolist()
        )

        print("\nVAL:")
        print(
            val_df["subject_id"].tolist()
        )

        print("\nTEST:")
        print(
            test_df["subject_id"].tolist()
        )

        print(
            f"\nSaved -> {out_path}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Get subjects
    # --------------------------------------------------------

    subject_ids = get_subject_ids(
        DATASET_PATH
    )

    print(
        f"Found {len(subject_ids)} subjects."
    )

    print(
        "\nSubject IDs:"
    )

    print(subject_ids)

    # --------------------------------------------------------
    # Get subject labels
    # --------------------------------------------------------

    subject_df = build_subject_label_dataframe(
        subject_ids
    )

    # --------------------------------------------------------
    # Print mapping
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SUBJECT LABELS")
    print("=" * 70)

    print(
        subject_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Generate folds
    # --------------------------------------------------------

    generate_splits(
        subject_df=subject_df,
        folds=N_FOLDS,
        out_dir=OUT_DIR,
        shuffle=SHUFFLE,
        seed=SEED,
    )