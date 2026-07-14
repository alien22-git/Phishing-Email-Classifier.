"""
make_plots.py
Generate the four report figures from saved metrics:
  1. PR curves: rules vs indicators-only vs TF-IDF+indicators
  2. Precision/recall vs threshold (rule score and ML probability)
  3. Indicator prevalence: phishing vs ham means
  4. Confusion matrices at the reported operating points
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
metrics = json.load(open(R / "metrics.json"))
curves = json.load(open(R / "pr_curves.json"))

COLORS = {
    "rules_weighted_score": "#d62728",
    "indicators_only_logreg": "#2ca02c",
    "tfidf_logreg": "#9467bd",
    "tfidf_plus_indicators_logreg": "#1f77b4",
}
LABELS = {
    "rules_weighted_score": "Rule-based (weighted score)",
    "indicators_only_logreg": "LogReg: 28 indicators only",
    "tfidf_logreg": "LogReg: TF-IDF only",
    "tfidf_plus_indicators_logreg": "LogReg: TF-IDF + indicators",
}

# ---- 1. PR curves ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
summary = {m["model"]: m for m in metrics["summary"]}
for name, c in curves.items():
    ap = summary.get(name, {}).get("pr_auc", None)
    lbl = LABELS[name] + (f"  (AP={ap:.3f})" if ap else "")
    ax.plot(c["recall"], c["precision"], color=COLORS[name], label=lbl, lw=2)
ax.axhline(0.724, color="gray", ls=":", lw=1, label="Random (prevalence=0.724)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall: ML classifiers vs rule-based baseline")
ax.set_xlim(0, 1.02); ax.set_ylim(0.55, 1.02)
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(R / "pr_curves.png", dpi=150)

# ---- 2. Threshold sweeps --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, sweep, title, xlab in [
    (axes[0], metrics["rule_threshold_sweep"],
     "Rule-based score", "Score threshold"),
    (axes[1], metrics["ml_threshold_sweep"],
     "ML model (TF-IDF + indicators)", "Probability threshold"),
]:
    t = [r["threshold"] for r in sweep]
    ax.plot(t, [r["precision"] for r in sweep], "o-", label="Precision", lw=2)
    ax.plot(t, [r["recall"] for r in sweep], "s-", label="Recall", lw=2)
    ax.plot(t, [r["f1"] for r in sweep], "^--", label="F1", lw=1.5, alpha=0.7)
    ax.set_title(title); ax.set_xlabel(xlab); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(alpha=0.3)
axes[0].set_ylabel("Metric value")
fig.suptitle("Precision / recall trade-off under varying thresholds", y=1.02)
fig.tight_layout(); fig.savefig(R / "threshold_sweep.png", dpi=150,
                                bbox_inches="tight")

# ---- 3. Indicator prevalence ----------------------------------------------
prev = metrics["indicator_prevalence"]
# pick indicators with the biggest phish/ham contrast, show top 14
items = sorted(prev.items(),
               key=lambda kv: abs(kv[1]["phish_mean"] - kv[1]["ham_mean"]),
               reverse=True)[:14]
names = [k for k, _ in items][::-1]
ph = [prev[k]["phish_mean"] for k in names]
hm = [prev[k]["ham_mean"] for k in names]
ypos = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(ypos + 0.2, ph, height=0.4, color="#d62728", label="Phishing")
ax.barh(ypos - 0.2, hm, height=0.4, color="#2ca02c", label="Legitimate")
ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Mean value per email (test set)")
ax.set_title("Handcrafted indicators: prevalence by class")
ax.legend(); ax.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig(R / "indicator_prevalence.png", dpi=150)

# ---- 4. Confusion matrices --------------------------------------------------
show = ["rules_weighted_score", "tfidf_plus_indicators_logreg"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name in zip(axes, show):
    m = summary[name]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: legit", "Pred: phish"])
    ax.set_yticklabels(["True: legit", "True: phish"])
    ax.set_title(f"{LABELS[name]}\n(P={m['precision']:.3f}, R={m['recall']:.3f})",
                 fontsize=10)
fig.suptitle("Confusion matrices at reported operating points")
fig.tight_layout(); fig.savefig(R / "confusion_matrices.png", dpi=150)

print("Saved 4 figures to results/")
