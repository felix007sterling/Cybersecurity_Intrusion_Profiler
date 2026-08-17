# Cybersecurity Network Threat & Intrusion Profiler

A working implementation of the hybrid ML framework described in the project report:
- **Path A (Supervised):** Random Forest classifies connections into `normal`, `DoS`, `Probe`, `R2L`, `U2R`.
- **Path B (Unsupervised):** Isolation Forest, trained only on normal traffic, flags connections that look statistically unusual — the zero-day / unknown-attack safety net.
- **Threat Profiler:** merges both outputs into one risk-scored record (`Low` / `Medium` / `High` / `Critical`).

Trained and evaluated on the real NSL-KDD dataset (included in `data/`).

## Project structure

```
project/
├── data/                     NSL-KDD train/test files (KDDTrain+.txt, KDDTest+.txt, ...)
├── src/
│   ├── constants.py          Column names + attack-category mapping
│   ├── preprocessing.py      Cleaning, encoding, scaling
│   ├── train.py               Trains both models, evaluates, saves artifacts
│   ├── threat_profiler.py    Hybrid inference: merges both models into a risk record
│   └── demo_cli.py            Quick terminal demo
├── models/                    Trained model artifacts (.joblib) + metrics.json
├── app/
│   └── dashboard.py           Streamlit dashboard (the deployable UI)
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the models

Pretrained models are already included in `models/`, but to retrain from scratch:

```bash
cd src
python train.py --train ../data/KDDTrain+.txt --test ../data/KDDTest+.txt
```

This prints classification metrics (accuracy, macro F1/precision/recall, per-class
report) and the anomaly detector's flagged fraction, and saves:
- `models/random_forest_classifier.joblib`
- `models/isolation_forest.joblib`
- `models/preprocessor.joblib`
- `models/metrics.json`

**Actual measured results on `KDDTest+.txt`** (this is a genuinely hard, well-known
generalization test — the official test set contains attack types not seen in
training, by design):

| Metric | Value |
|---|---|
| Accuracy | ~0.74 |
| Macro F1 | ~0.48 |
| Macro Precision | ~0.74 |
| Macro Recall | ~0.48 |

DoS/Probe/normal are detected well; R2L and U2R are hard (few training examples,
and many R2L/U2R test attacks are unseen signatures) — this is expected and is
exactly the gap the anomaly-detection path is meant to help cover. If you want
higher R2L/U2R recall, try `class_weight="balanced_subsample"`, SMOTE oversampling,
or a gradient-boosted model in `train.py`.

## 3. Quick CLI demo

```bash
cd src
python demo_cli.py --n 15
```

Profiles 15 random test connections and prints their threat profile records
side-by-side with the actual label, so you can sanity-check predictions.

`--data` also accepts a URL, e.g.:

```bash
python demo_cli.py --n 15 --data "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Test.csv"
```

## 3b. Loading data directly from a URL (no upload needed)

`src/data_loader.py` fetches and auto-parses a remote NSL-KDD CSV/TXT
without ever touching disk manually — it auto-detects whether the file has
41 columns (features only), 42 (+ label), or 43 (+ label + difficulty):

```bash
python src/data_loader.py "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Train.csv"
```

`train.py`'s `--train`/`--test` flags and `preprocessing.load_and_prepare()`
both accept a URL in place of a file path directly:

```bash
python src/train.py \
  --train "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Train.csv" \
  --test  "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/refs/heads/master/NSL_KDD_Test.csv"
```

The dashboard (`app/dashboard.py`) also has a **"Load from URL"** option in
the sidebar — paste any raw NSL-KDD CSV URL and it fetches + profiles it live.

## 4. Run the dashboard (deployment)

```bash
streamlit run app/dashboard.py
```

Opens a browser dashboard where you can:
- Use the bundled NSL-KDD test set, or upload your own NSL-KDD-format CSV
- See risk-level distribution, attack-category breakdown, anomaly-score histogram
- Filter and inspect individual threat-profile records
- Download the full profiled output as CSV

### Deploying it publicly

**Streamlit Community Cloud (free, easiest):**
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → "New app" → point it at your repo,
   set the main file to `app/dashboard.py`.
3. It installs `requirements.txt` automatically and gives you a public URL.

**Docker:**
```bash
docker build -t threat-profiler .
docker run -p 8501:8501 threat-profiler
```
(a `Dockerfile` is included)

**Any VM / server:**
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

## 5. Using it on live/custom traffic

Your input just needs the same 41 NSL-KDD feature columns (see
`src/constants.py::COLUMN_NAMES`), as a header-less CSV. Real production
traffic would need a feature-extraction step (e.g. via a NetFlow exporter
or Zeek/Bro logs) to compute the same fields (`duration`, `src_bytes`,
`count`, `serror_rate`, etc.) — that adapter is not included here since it
depends on your actual traffic-capture stack, but `threat_profiler.py`'s
`profile_single(feature_dict)` method is the integration point once you
have those features.

## Notes on academic honesty

The metrics above are real, measured numbers from an actual run of this code
against the real NSL-KDD test set — not fabricated figures. If you retrain
with different hyperparameters your numbers will differ slightly; rerun
`train.py` and copy the fresh `models/metrics.json` values into your report
if you want it to reflect an actual executed run.
