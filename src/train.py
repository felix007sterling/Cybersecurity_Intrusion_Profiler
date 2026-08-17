"""
Trains the two models described in the report:

  Path A (supervised)   -> RandomForestClassifier over
                            {normal, DoS, Probe, R2L, U2R}
  Path B (unsupervised)  -> IsolationForest fit on normal traffic only,
                            used as the zero-day / anomaly signal.

Run:
    python train.py --train ../data/KDDTrain+.txt --test ../data/KDDTest+.txt
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score, roc_auc_score
)
import joblib

from preprocessing import load_and_prepare, NSLKDDPreprocessor
from constants import ATTACK_CATEGORIES


def train(train_path, test_path, out_dir, n_estimators=200, contamination=0.1):
    os.makedirs(out_dir, exist_ok=True)

    print("[1/6] Loading & cleaning training data...")
    train_df = load_and_prepare(train_path)
    test_df = load_and_prepare(test_path)

    print("[2/6] Fitting preprocessor (encoding + scaling)...")
    pre = NSLKDDPreprocessor()
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)

    y_train = train_df["attack_category"].values
    y_test = test_df["attack_category"].values

    print("[3/6] Training Random Forest classifier (Path A: known attacks)...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    print("[4/6] Training Isolation Forest anomaly detector (Path B: zero-day)...")
    normal_mask = y_train == "normal"
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train[normal_mask])

    print("[5/6] Evaluating on held-out test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    anomaly_scores = iso.decision_function(X_test)   # higher = more normal
    anomaly_pred = iso.predict(X_test)                # -1 = anomaly, 1 = normal
    # Treat "actual attack" as the anomaly-relevant ground truth for a
    # sanity-check AUC (not the primary metric, since IF is unsupervised).
    y_bin_attack = (y_test != "normal").astype(int)
    # decision_function: lower score = more anomalous, so invert for AUC vs attack=1
    try:
        auc = roc_auc_score(y_bin_attack, -anomaly_scores)
    except ValueError:
        auc = None

    metrics = {
        "classifier": {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "per_class_report": report,
            "confusion_matrix": cm.tolist(),
            "classes": list(clf.classes_),
        },
        "anomaly_detector": {
            "contamination": contamination,
            "flagged_fraction": float((anomaly_pred == -1).mean()),
            "sanity_check_auc_vs_actual_attack_label": auc,
        },
    }

    print("[6/6] Saving artifacts...")
    joblib.dump(clf, os.path.join(out_dir, "random_forest_classifier.joblib"))
    joblib.dump(iso, os.path.join(out_dir, "isolation_forest.joblib"))
    pre.save(os.path.join(out_dir, "preprocessor.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n=== SUPERVISED CLASSIFIER (Random Forest) ===")
    print(f"Accuracy:        {metrics['classifier']['accuracy']:.4f}")
    print(f"Macro F1:        {metrics['classifier']['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['classifier']['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['classifier']['macro_recall']:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== ANOMALY DETECTOR (Isolation Forest) ===")
    print(f"Flagged as anomalous: {metrics['anomaly_detector']['flagged_fraction']*100:.2f}% of test traffic")
    if auc is not None:
        print(f"Sanity-check AUC (anomaly score vs actual attack label): {auc:.4f}")
    print(f"\nAll artifacts saved to: {out_dir}")

    return clf, iso, pre, metrics


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Train the hybrid classifier + anomaly detector. "
                     "--train/--test accept either a local file path or an "
                     "http(s):// URL to a raw NSL-KDD CSV."
    )
    parser.add_argument("--train", default=os.path.join(here, "..", "data", "KDDTrain+.txt"))
    parser.add_argument("--test", default=os.path.join(here, "..", "data", "KDDTest+.txt"))
    parser.add_argument("--out", default=os.path.join(here, "..", "models"))
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--contamination", type=float, default=0.1)
    args = parser.parse_args()

    train(args.train, args.test, args.out, args.n_estimators, args.contamination)
