"""
Data pipeline for the dropout project.

These are the Milestone 2 notebook cells (notebooks/milestone2_data_approval.ipynb)
turned into functions so every notebook and script builds the exact same design
matrix and split. No cleaning decision was changed from M2; the section numbers in
the comments point back to the M2 notebook.
"""
import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42   # same seed as Lab 2 / Lab 3 and Milestone 2

# paths are relative to the repo root, so this works on every team member's machine
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "data.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
DATA_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"   # M2 section 2.1

CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]
CLASS_COLORS = {"Dropout": "#eb6834", "Enrolled": "#2a78d6", "Graduate": "#1baf7a"}
target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}

# M2 section 5: every column except Target is in exactly one of these groups
categorical_cols = ["Marital status", "Application mode", "Course", "Previous qualification",
                    "Mother's qualification", "Father's qualification",
                    "Mother's occupation", "Father's occupation"]

binary_cols = ["Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
               "Tuition fees up to date", "Gender", "Scholarship holder", "International",
               "no_units_1st_sem"]

continuous_cols = ["Application order", "Previous qualification (grade)", "Admission grade",
                   "Age at enrollment",
                   "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
                   "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
                   "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
                   "Unemployment rate", "Inflation rate", "GDP"]

second_sem_cols = ["Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
                   "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
                   "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)"]


def file_sha256(path):
    """
    Parameters:
        `path` - path to a file
    Returns: the SHA-256 hex digest of the file's bytes
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_raw(path=DATA_PATH):
    """
    Load the raw UCI #697 csv into a DataFrame, downloading it first if it's missing.

    Parameters:
        `path` - where data.csv lives (default: data/data.csv in the repo)
    Returns: the raw DataFrame, shape (4424, 37)
    """
    path = Path(path)
    if not path.exists():
        # same download as M2 section 2, but extracted straight into data/
        path.parent.mkdir(parents=True, exist_ok=True)
        zip_path = path.parent / "student_dropout.zip"
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("data.csv", path.parent)
        os.remove(zip_path)

    # a different hash means a different dataset version, so stop instead of
    # silently training on it
    assert file_sha256(path) == DATA_SHA256, "data.csv does not match the M2 dataset version"

    # the file separates values with ';' and starts with a byte-order mark
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def occupation_group(code):
    """
    Collapse a 3-digit occupation code to its major group from the data dictionary (M2 3.6).

    Parameters:
        `code` - an integer occupation code
    Returns: the major-group code (0-10, 90 other, 99 blank)
    """
    if code < 100:            # 0-10, 90 (other), 99 (blank) are already major groups
        return code
    group = (code // 10) % 10  # middle digit: 122 -> 2, 175 -> 7
    return 10 if group == 0 else group   # 101-103 -> 10 Armed Forces


def clean(df, keep_second_sem=False):
    """
    Apply the M2 cleaning steps (the five kinds of bad data, Week 2).

    Parameters:
        `df` - the raw DataFrame from load_raw()
        `keep_second_sem` - keep the six 2nd-semester columns. False for our
                            early-warning task (M2 3.5); True only for the
                            sensitivity run that matches the published setup
    Returns: a cleaned copy of `df`
    """
    df = df.copy()

    # 3.1 junk values: a column name ends in a tab, and one is misspelled
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Nacionality": "Nationality"})

    # 3.2 redundant information: duplicate rows (none in this version) and the
    # Nationality column, which gives the same answer as International
    df = df.drop_duplicates()
    df = df.drop(columns="Nationality")

    # 3.4 contextual errors: 0 enrolled units is a recording gap (Course 171), not "failed everything"
    df["no_units_1st_sem"] = (df["Curricular units 1st sem (enrolled)"] == 0).astype(int)

    # 3.5 irrelevant information: the 2nd semester happens after our prediction point
    if not keep_second_sem:
        df = df.drop(columns=second_sem_cols)

    # 3.6 sparse occupation codes -> 10 major groups
    for col in ["Mother's occupation", "Father's occupation"]:
        df[col] = df[col].map(occupation_group)

    return df


def build_design_matrix(df):
    """
    One-hot encode the categorical codes and stack everything into the design matrix (Week 2).

    Parameters:
        `df` - a cleaned DataFrame from clean()
    Returns: (X, t) where X is the design matrix (one row per student) and t holds
             the integer labels 0 = Dropout, 1 = Enrolled, 2 = Graduate
    """
    num_cols = continuous_cols + [c for c in second_sem_cols if c in df.columns]

    # get_dummies only decides which columns exist, it doesn't learn statistics
    # from the data, so running it before the split can't leak anything (M2 5)
    dummies = pd.get_dummies(df[categorical_cols].astype(str), dtype=int)
    X = pd.concat([df[num_cols + binary_cols], dummies], axis="columns")
    t = df["Target"].map(target_map)
    return X, t


def split_data(X, t, seed=RANDOM_STATE):
    """
    Stratified 70/15/15 split, the same two train_test_split calls as M2 section 8.

    Parameters:
        `X` - the design matrix
        `t` - the labels
        `seed` - random_state for both calls (42 is the official split)
    Returns: X_train, X_valid, X_test, t_train, t_valid, t_test
    """
    # first set aside 30%, then split that 30% in half
    X_train, X_rest, t_train, t_rest = train_test_split(
        X, t, test_size=0.3, stratify=t, random_state=seed)
    X_valid, X_test, t_valid, t_test = train_test_split(
        X_rest, t_rest, test_size=0.5, stratify=t_rest, random_state=seed)
    return X_train, X_valid, X_test, t_train, t_valid, t_test


def normalize(X_train, X_valid, X_test, cols, epsilon=0.0001):
    """
    Normalize `cols` to mean 0 / std 1 using statistics from the training set only (Lab 3, M2 8.2).

    Parameters:
        `X_train`, `X_valid`, `X_test` - the three splits
        `cols` - the continuous columns to rescale (binary/one-hot columns stay 0/1)
        `epsilon` - avoids dividing by a standard deviation of 0
    Returns: normalized copies (X_train_norm, X_valid_norm, X_test_norm)
    """
    # the normalization parameters are part of the model, so they come from training data only
    X_mean = X_train[cols].mean(axis=0)
    X_std = X_train[cols].std(axis=0, ddof=0)

    X_train_norm, X_valid_norm, X_test_norm = X_train.copy(), X_valid.copy(), X_test.copy()
    X_train_norm[cols] = (X_train[cols] - X_mean) / (X_std + epsilon)
    X_valid_norm[cols] = (X_valid[cols] - X_mean) / (X_std + epsilon)
    X_test_norm[cols] = (X_test[cols] - X_mean) / (X_std + epsilon)
    return X_train_norm, X_valid_norm, X_test_norm


def check_split(X, X_train, X_valid, X_test):
    """
    The M2 split-integrity and leakage checks, as asserts.

    Parameters:
        `X` - the full design matrix
        `X_train`, `X_valid`, `X_test` - the three splits
    Returns: nothing; raises AssertionError if a check fails
    """
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_valid.index).isdisjoint(X_test.index)
    assert list(X_train.columns) == list(X_valid.columns) == list(X_test.columns)
    assert X_train.isnull().sum().sum() == 0
    assert X_valid.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0


def load_splits(seed=RANDOM_STATE, keep_second_sem=False):
    """
    Run the whole pipeline: load -> clean -> design matrix -> split -> checks -> normalize.

    Parameters:
        `seed` - split seed (42 is the official split)
        `keep_second_sem` - see clean()
    Returns: a dict with the normalized splits, the labels, the raw (un-normalized)
             splits' row indices, and the list of normalized columns
    """
    df = clean(load_raw(), keep_second_sem=keep_second_sem)
    X, t = build_design_matrix(df)
    X_train, X_valid, X_test, t_train, t_valid, t_test = split_data(X, t, seed=seed)
    check_split(X, X_train, X_valid, X_test)
    if not keep_second_sem:
        assert not any("2nd sem" in c for c in X_train.columns)

    norm_cols = continuous_cols + [c for c in second_sem_cols if c in X.columns]
    X_train_norm, X_valid_norm, X_test_norm = normalize(X_train, X_valid, X_test, norm_cols)

    return {
        "X": X, "t": t,
        "X_train": X_train_norm, "X_valid": X_valid_norm, "X_test": X_test_norm,
        "t_train": t_train, "t_valid": t_valid, "t_test": t_test,
        "norm_cols": norm_cols,
    }
