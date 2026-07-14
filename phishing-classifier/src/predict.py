"""
predict.py
Classify a single raw email file (.eml or SpamAssassin format) and explain
which indicators fired.

Usage:
    python src/predict.py path/to/email.eml [--model models/phishing_classifier.joblib]
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from features import build_feature_frame
from parse_emails import parse_file
from rules import RuleScorer
from train import clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("email_path")
    ap.add_argument("--model", default="models/phishing_classifier.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    row = parse_file(Path(args.email_path), label=-1, source="cli")
    if row is None:
        sys.exit("Could not parse email file.")
    df = pd.DataFrame([row])

    bundle = joblib.load(args.model)
    vec, scaler, model = bundle["vectorizer"], bundle["scaler"], bundle["model"]
    cols = bundle["feature_columns"]

    feats = build_feature_frame(df)[cols]
    Xt = vec.transform([clean_text(row["subject"], row["body"])])
    Xf = csr_matrix(scaler.transform(feats))
    prob = float(model.predict_proba(hstack([Xt, Xf]))[0, 1])

    verdict = "PHISHING" if prob >= args.threshold else "LEGITIMATE"
    print(f"\nFile:     {args.email_path}")
    print(f"Subject:  {row['subject'][:80]}")
    print(f"From:     {row['from_raw'][:80]}")
    print(f"\nVerdict:  {verdict}  (score={prob:.4f}, threshold={args.threshold})")
    print(f"Rule score: {RuleScorer().score(feats)[0]:.1f}")

    fired = feats.iloc[0]
    fired = fired[fired > 0].sort_values(ascending=False)
    if len(fired):
        print("\nIndicators fired:")
        for name, val in fired.items():
            print(f"  {name:<28} {val:g}")
    else:
        print("\nNo handcrafted indicators fired (verdict driven by text model).")


if __name__ == "__main__":
    main()
