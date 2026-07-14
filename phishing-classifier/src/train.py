"""
train.py
Train and benchmark four classifiers on the same stratified train/test split:

  1. AnyTwoRules        - binary heuristic (single operating point)
  2. RuleScorer         - weighted rule score (threshold sweepable)
  3. TF-IDF + LogisticRegression        (text only)
  4. TF-IDF + indicators + LogisticRegression  (text + handcrafted features)

Outputs metrics.json, threshold sweep tables, PR/ROC data, and saved models.

Usage:
    python src/train.py --data data/processed/emails.parquet \
        --features data/processed/features.parquet --outdir results
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import rules as rules_mod

SEED = 42


def clean_text(subject, body):
    """Subject + body, lowercased, URLs and long tokens collapsed to
    placeholders so TF-IDF learns language, not memorized links.
    Also strips digits so it can't key on years/dates (the ham corpus is
    older than the phishing corpus)."""
    text = f"{subject or ''} {body or ''}".lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:20000]


def sweep_thresholds(y_true, scores, points):
    rows = []
    for t in points:
        pred = (scores >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 4),
            "precision": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, pred, zero_division=0), 4),
            "flagged_pct": round(100.0 * pred.mean(), 2),
        })
    return rows


def summarize(name, y_true, scores=None, preds=None, threshold=None):
    out = {"model": name}
    if scores is not None:
        out["roc_auc"] = round(roc_auc_score(y_true, scores), 4)
        out["pr_auc"] = round(average_precision_score(y_true, scores), 4)
        if threshold is None:
            # F1-optimal threshold on the test set is reported for reference
            prec, rec, thr = precision_recall_curve(y_true, scores)
            f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            threshold = float(thr[np.argmax(f1s[:-1])])
        preds = (scores >= threshold).astype(int)
        out["threshold"] = round(float(threshold), 4)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    out.update({
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--modeldir", default="models")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    modeldir = Path(args.modeldir); modeldir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    feats = pd.read_parquet(args.features).drop(columns=["label"])
    y = df["label"].values
    text = [clean_text(s, b) for s, b in zip(df["subject"], df["body"])]

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.25, stratify=y, random_state=SEED)
    y_tr, y_te = y[idx_train], y[idx_test]
    feats_tr, feats_te = feats.iloc[idx_train], feats.iloc[idx_test]

    print(f"Train: {len(idx_train)} ({y_tr.mean():.1%} phish) | "
          f"Test: {len(idx_test)} ({y_te.mean():.1%} phish)")

    results, curves = [], {}

    # ---- Baseline 1: any-two-families heuristic -------------------------
    b1 = rules_mod.AnyTwoRules(k=2)
    results.append(summarize("rules_any2families", y_te,
                             preds=b1.predict(feats_te)))

    # ---- Baseline 2: weighted rule score --------------------------------
    # Fixed operating threshold. Note: the F1-"optimal" threshold on this
    # test set degenerates to 0 (flag everything scores F1=0.84 because 72%
    # of the test set is phishing), so we report a realistic SOC-style
    # operating point instead and provide the full sweep table.
    b2 = rules_mod.RuleScorer()
    s_rules = b2.score(feats_te)
    results.append(summarize("rules_weighted_score", y_te, scores=s_rules,
                             threshold=2.0))
    prec, rec, _ = precision_recall_curve(y_te, s_rules)
    curves["rules_weighted_score"] = {"precision": prec.tolist(),
                                      "recall": rec.tolist()}
    rule_sweep = sweep_thresholds(
        y_te, s_rules, np.arange(1, 13, 1))

    # ---- Model 3: TF-IDF only -------------------------------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                          sublinear_tf=True, strip_accents="unicode")
    Xt_tr = vec.fit_transform([text[i] for i in idx_train])
    Xt_te = vec.transform([text[i] for i in idx_test])

    clf_text = LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight="balanced", random_state=SEED)
    clf_text.fit(Xt_tr, y_tr)
    p_text = clf_text.predict_proba(Xt_te)[:, 1]
    results.append(summarize("tfidf_logreg", y_te, scores=p_text,
                             threshold=0.5))
    prec, rec, _ = precision_recall_curve(y_te, p_text)
    curves["tfidf_logreg"] = {"precision": prec.tolist(),
                              "recall": rec.tolist()}

    # ---- Model 3b: handcrafted indicators only (ablation) ----------------
    # Era-robust-ish comparison: no vocabulary, only the 28 engineered
    # indicators. Shows how much of the ML lift comes from text.
    scaler_only = StandardScaler()
    Xi_tr = scaler_only.fit_transform(feats_tr)
    Xi_te = scaler_only.transform(feats_te)
    clf_ind = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=SEED)
    clf_ind.fit(Xi_tr, y_tr)
    p_ind = clf_ind.predict_proba(Xi_te)[:, 1]
    results.append(summarize("indicators_only_logreg", y_te, scores=p_ind,
                             threshold=0.5))
    prec, rec, _ = precision_recall_curve(y_te, p_ind)
    curves["indicators_only_logreg"] = {"precision": prec.tolist(),
                                        "recall": rec.tolist()}

    # ---- Model 4: TF-IDF + handcrafted indicators -----------------------
    scaler = StandardScaler()
    Xf_tr = csr_matrix(scaler.fit_transform(feats_tr))
    Xf_te = csr_matrix(scaler.transform(feats_te))
    Xc_tr, Xc_te = hstack([Xt_tr, Xf_tr]), hstack([Xt_te, Xf_te])

    clf_comb = LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight="balanced", random_state=SEED)
    clf_comb.fit(Xc_tr, y_tr)
    p_comb = clf_comb.predict_proba(Xc_te)[:, 1]
    results.append(summarize("tfidf_plus_indicators_logreg", y_te,
                             scores=p_comb, threshold=0.5))
    prec, rec, _ = precision_recall_curve(y_te, p_comb)
    curves["tfidf_plus_indicators_logreg"] = {"precision": prec.tolist(),
                                              "recall": rec.tolist()}
    ml_sweep = sweep_thresholds(
        y_te, p_comb, [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99])

    # ---- ROC data for the two score-based headliners --------------------
    roc = {}
    for name, s in [("rules_weighted_score", s_rules),
                    ("tfidf_plus_indicators_logreg", p_comb)]:
        fpr, tpr, _ = roc_curve(y_te, s)
        roc[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # ---- Top indicative n-grams (interpretability) ----------------------
    vocab = np.array(vec.get_feature_names_out())
    coefs = clf_text.coef_[0]
    top_phish = vocab[np.argsort(coefs)[-25:]][::-1].tolist()
    top_ham = vocab[np.argsort(coefs)[:25]].tolist()

    # ---- Indicator prevalence by class (for the report) ------------------
    prevalence = {}
    for col in feats.columns:
        prevalence[col] = {
            "phish_mean": round(float(feats_te[y_te == 1][col].mean()), 3),
            "ham_mean": round(float(feats_te[y_te == 0][col].mean()), 3),
        }

    # ---- Error analysis samples ------------------------------------------
    test_df = df.iloc[idx_test]
    fp_ids = test_df[(y_te == 0) & (p_comb >= 0.5)]["id"].tolist()[:15]
    fn_ids = test_df[(y_te == 1) & (p_comb < 0.5)]["id"].tolist()[:15]

    # ---- Save -------------------------------------------------------------
    json.dump({
        "summary": results,
        "rule_threshold_sweep": rule_sweep,
        "ml_threshold_sweep": ml_sweep,
        "top_phish_ngrams": top_phish,
        "top_ham_ngrams": top_ham,
        "indicator_prevalence": prevalence,
        "false_positive_ids": fp_ids,
        "false_negative_ids": fn_ids,
        "n_train": int(len(idx_train)), "n_test": int(len(idx_test)),
    }, open(outdir / "metrics.json", "w"), indent=2)
    json.dump(curves, open(outdir / "pr_curves.json", "w"))
    json.dump(roc, open(outdir / "roc_curves.json", "w"))

    joblib.dump({"vectorizer": vec, "scaler": scaler, "model": clf_comb,
                 "feature_columns": list(feats.columns)},
                modeldir / "phishing_classifier.joblib")

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved metrics -> {outdir}/metrics.json, "
          f"model -> {modeldir}/phishing_classifier.joblib")


if __name__ == "__main__":
    main()
