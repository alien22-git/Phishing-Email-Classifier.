"""
ood_probe.py
Out-of-distribution probe.

Problem: the ham corpus (SpamAssassin, 2002) and the phishing corpus
(phishing_pot, 2022+) come from different eras. A classifier could score
well by learning "old mailing-list email vs modern email" instead of
"legitimate vs phishing".

Probe: run the trained model on SpamAssassin spam_2, which is 2002-era
*malicious* email (scams, credential lures, pump-and-dumps). It shares the
era and formatting of the ham corpus but the intent of the phishing corpus.

  - If the model keyed mostly on era artifacts -> it will call 2002 spam
    "legitimate" (low flag rate).
  - If it learned content signals -> it will flag a substantial share.

This does not fully disentangle the confound (spam != phishing, so 100% is
not the target), but a very low flag rate is a red flag for artifact
learning.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from features import build_feature_frame
from parse_emails import collect
from train import clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spamdir", required=True)
    ap.add_argument("--model", default="models/phishing_classifier.joblib")
    ap.add_argument("--out", default="results/ood_probe.json")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    vec, scaler, model = bundle["vectorizer"], bundle["scaler"], bundle["model"]
    cols = bundle["feature_columns"]

    rows = collect([args.spamdir], 1, "ood_spam")
    df = pd.DataFrame(rows).drop_duplicates(subset=["subject", "body"])
    print(f"Loaded {len(df)} spam_2 emails (2002-era malicious mail)")

    text = [clean_text(s, b) for s, b in zip(df["subject"], df["body"])]
    Xt = vec.transform(text)
    feats = build_feature_frame(df)[cols]
    Xf = csr_matrix(scaler.transform(feats))
    p = model.predict_proba(hstack([Xt, Xf]))[:, 1]

    out = {
        "n_emails": int(len(df)),
        "flag_rate_at_0.5": round(float((p >= 0.5).mean()), 4),
        "flag_rate_at_0.3": round(float((p >= 0.3).mean()), 4),
        "flag_rate_at_0.7": round(float((p >= 0.7).mean()), 4),
        "mean_score": round(float(p.mean()), 4),
        "median_score": round(float(np.median(p)), 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
