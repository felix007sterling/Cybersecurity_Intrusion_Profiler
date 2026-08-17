"""
Quick command-line demo: profiles N random connections from the test set
and prints the resulting threat profiles as a table.

    python src/demo_cli.py --n 15
"""
import os
import argparse
import pandas as pd

from preprocessing import load_and_prepare
from threat_profiler import ThreatProfiler

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--model_dir", default=os.path.join(here, "..", "models"))
    parser.add_argument("--data", default=os.path.join(here, "..", "data", "KDDTest+.txt"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    df = load_and_prepare(args.data)
    sample = df.sample(args.n, random_state=args.seed).reset_index(drop=True)
    true_labels = sample["attack_category"].tolist()

    profiler = ThreatProfiler(args.model_dir)
    profiles = profiler.profile_batch(sample)
    profiles.insert(2, "actual_category", true_labels)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print(profiles.to_string(index=False))

    n_critical = (profiles["risk_level"] == "Critical").sum()
    n_high = (profiles["risk_level"] == "High").sum()
    print(f"\n{n_critical} Critical, {n_high} High risk connection(s) flagged out of {len(profiles)}.")
