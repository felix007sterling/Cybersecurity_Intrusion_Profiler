"""
Data loading and preprocessing for the NSL-KDD dataset.

Implements the pipeline described in Section 9 of the project report:
missing/invalid value handling, duplicate removal, categorical encoding,
numerical scaling, and train/test splitting.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from constants import COLUMN_NAMES, CATEGORICAL_COLS, ATTACK_CATEGORY_MAP


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw NSL-KDD .txt (CSV, no header) file into a DataFrame."""
    df = pd.read_csv(path, names=COLUMN_NAMES, header=None)
    return df


def load_raw_source(source: str) -> pd.DataFrame:
    """
    Load NSL-KDD traffic from either a local file path or a remote URL
    (http:// or https://). Auto-detects the 41/42/43-column format when
    loading from a URL. Local paths are still assumed to be the full
    43-column KDDTrain+/KDDTest+ format.
    """
    if source.startswith("http://") or source.startswith("https://"):
        from data_loader import load_dataset_from_url
        return load_dataset_from_url(source)
    return load_raw(source)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value handling and duplicate removal."""
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def add_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the raw 'label' column (specific signature) to a broad category.
    If no 'label' column is present (e.g. unlabeled live/raw traffic),
    this is a no-op and attack_category is left unset.
    """
    df = df.copy()
    if "label" not in df.columns:
        return df
    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP)
    # Any signature not in our map (rare/unseen) is treated as 'unknown'
    # and folded into the nearest generic bucket rather than dropped.
    df["attack_category"] = df["attack_category"].fillna("unknown")
    return df


class NSLKDDPreprocessor:
    """
    Fits categorical encoders + a scaler on training data, and applies
    the identical transform to any future dataframe (test data, or a
    single new connection record at inference time).
    """

    def __init__(self):
        self.category_maps = {}   # col -> {category: code}
        self.feature_cols = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.feature_cols = [c for c in COLUMN_NAMES if c not in ("label", "difficulty")]

        for col in CATEGORICAL_COLS:
            uniques = sorted(df[col].astype(str).unique())
            self.category_maps[col] = {v: i for i, v in enumerate(uniques)}
            # Reserve an "unseen" bucket at the end
            self.category_maps[col]["__unseen__"] = len(uniques)

        encoded = self._encode_categoricals(df)
        self.scaler.fit(encoded[self.feature_cols].values)
        self.fitted = True
        return self

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            mapping = self.category_maps[col]
            df[col] = df[col].astype(str).map(lambda v: mapping.get(v, mapping["__unseen__"]))
        return df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        encoded = self._encode_categoricals(df)
        X = self.scaler.transform(encoded[self.feature_cols].values)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "NSLKDDPreprocessor":
        return joblib.load(path)


def load_and_prepare(source: str):
    """
    Convenience: load raw file-or-URL -> clean -> attach attack_category.
    `source` may be a local file path or an http(s):// URL.
    """
    df = load_raw_source(source)
    df = clean(df)
    df = add_attack_category(df)
    return df


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(here, "..", "data", "KDDTrain+.txt")
    df = load_and_prepare(train_path)
    print(df["attack_category"].value_counts())
    print(df.shape)
