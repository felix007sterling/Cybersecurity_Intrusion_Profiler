"""
Load NSL-KDD-format traffic directly from a URL (e.g. a raw GitHub CSV),
with no local file upload required.

Handles the three column-count variants seen in the wild:
  - 43 cols: 41 features + label + difficulty   (original KDDTrain+.txt)
  - 42 cols: 41 features + label                 (e.g. the Mamcose repo CSV)
  - 41 cols: 41 features only, no label           (raw/unlabeled traffic)
"""
import io
import requests
import pandas as pd

from constants import COLUMN_NAMES

FEATURE_COLS = COLUMN_NAMES[:-2]   # the 41 feature names, no label/difficulty


def fetch_csv_text(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_nslkdd_csv(text: str) -> pd.DataFrame:
    """
    Auto-detects whether the CSV has 41, 42, or 43 columns per row and
    assigns the correct column names accordingly.
    """
    first_line = text.strip().splitlines()[0]
    n_cols = len(first_line.split(","))

    if n_cols == 43:
        names = COLUMN_NAMES
    elif n_cols == 42:
        names = FEATURE_COLS + ["label"]
    elif n_cols == 41:
        names = FEATURE_COLS
    else:
        raise ValueError(
            f"Unrecognized NSL-KDD CSV format: found {n_cols} columns per row "
            f"(expected 41, 42, or 43)."
        )

    df = pd.read_csv(io.StringIO(text), names=names, header=None)
    return df


def load_dataset_from_url(url: str) -> pd.DataFrame:
    """Fetch + parse a remote NSL-KDD CSV/txt file into a DataFrame."""
    text = fetch_csv_text(url)
    return parse_nslkdd_csv(text)


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else (
        "https://raw.githubusercontent.com/Mamcose/"
        "NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Train.csv"
    )
    df = load_dataset_from_url(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from:\n  {url}")
    print(df.head())
    if "label" in df.columns:
        print("\nLabel counts:")
        print(df["label"].value_counts().head(10))
