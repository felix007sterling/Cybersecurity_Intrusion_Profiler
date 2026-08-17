"""
Hybrid Threat Detection Architecture (Section 14-15 of the report).

Combines:
  - Random Forest category prediction + confidence
  - Isolation Forest anomaly score
into a single threat-profile record per connection, with a risk level.
"""
import os
import uuid
import datetime
import numpy as np
import pandas as pd
import joblib

from preprocessing import NSLKDDPreprocessor


class ThreatProfiler:
    def __init__(self, model_dir: str):
        self.clf = joblib.load(os.path.join(model_dir, "random_forest_classifier.joblib"))
        self.iso = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
        self.pre = NSLKDDPreprocessor.load(os.path.join(model_dir, "preprocessor.joblib"))

    def _risk_level(self, category: str, class_confidence: float, anomaly_score: float, is_anomaly: bool) -> str:
        """
        Combines the two signals into one of Low / Medium / High / Critical.

        Rules (documented, deterministic — mirrors Table 10 in the report):
          - normal + not anomalous                       -> Low
          - normal + anomalous (score says unusual)        -> Medium  (possible zero-day)
          - known attack, low confidence                   -> Medium
          - known attack, high confidence, not anomalous    -> High
          - known attack, high confidence, AND anomalous     -> Critical
        """
        if category == "normal":
            return "Medium" if is_anomaly else "Low"

        # known attack category
        if is_anomaly and class_confidence >= 0.6:
            return "Critical"
        if class_confidence >= 0.6:
            return "High"
        return "Medium"

    def profile_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: raw-format dataframe with the 41 NSL-KDD feature columns
            (label/difficulty/attack_category columns are ignored if present).
        Returns a DataFrame of threat-profile records.
        """
        X = self.pre.transform(df)

        pred_category = self.clf.predict(X)
        pred_proba = self.clf.predict_proba(X)
        class_confidence = pred_proba.max(axis=1)

        anomaly_raw = self.iso.decision_function(X)     # higher = more normal
        anomaly_flag = self.iso.predict(X)                # -1 anomaly, 1 normal
        # normalize anomaly_raw into a 0-100 "suspicion score" (higher = more suspicious)
        lo, hi = anomaly_raw.min(), anomaly_raw.max()
        span = (hi - lo) if hi > lo else 1.0
        suspicion_score = 100 * (1 - (anomaly_raw - lo) / span)

        records = []
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for i in range(len(df)):
            is_anom = anomaly_flag[i] == -1
            cat = pred_category[i]
            conf = float(class_confidence[i])
            risk = self._risk_level(cat, conf, float(suspicion_score[i]), is_anom)
            records.append({
                "threat_id": str(uuid.uuid4())[:8],
                "timestamp": now,
                "predicted_attack_type": cat,
                "classification_confidence": round(conf, 4),
                "anomaly_score": round(float(suspicion_score[i]), 2),
                "is_anomalous": bool(is_anom),
                "risk_level": risk,
                "protocol_type": df.iloc[i].get("protocol_type", None),
                "service": df.iloc[i].get("service", None),
                "flag": df.iloc[i].get("flag", None),
            })
        return pd.DataFrame(records)

    def profile_single(self, feature_dict: dict) -> dict:
        """Profile a single connection given as a dict of the 41 raw features."""
        df = pd.DataFrame([feature_dict])
        return self.profile_batch(df).iloc[0].to_dict()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "..", "models")
    test_path = os.path.join(here, "..", "data", "KDDTest+.txt")

    from preprocessing import load_and_prepare
    df = load_and_prepare(test_path).sample(10, random_state=1).reset_index(drop=True)

    profiler = ThreatProfiler(model_dir)
    profiles = profiler.profile_batch(df)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(profiles)
