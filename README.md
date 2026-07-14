# Phishing Email Classifier

Detects phishing emails from raw `.eml` files using TF-IDF text features plus 28
handcrafted indicators across three families: **urgency language**, **suspicious
URL characteristics**, and **sender inconsistencies** (header analysis).
Benchmarked against two rule-based baselines with a full precision/recall
trade-off analysis under varying thresholds.

## Data

Real emails, full headers (headers are required for sender-inconsistency features,
which most CSV phishing datasets strip out):

| Source | Class | Count (after dedup) |
|---|---|---|
| [phishing_pot](https://github.com/rf-peixoto/phishing_pot) | Phishing (real samples, 2022+) | 6,920 |
| [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) `easy_ham` + `hard_ham` | Legitimate | 2,640 |

11,410 raw emails parsed; exact subject+body duplicates dropped (phishing kits
resend identical mail, and duplicates across a random split leak training data
into the test set). Stratified 75/25 train/test split.

```bash
bash data/download_data.sh   # fetches both corpora
```

## Pipeline

```
parse_emails.py  ->  features.py  ->  train.py  ->  make_plots.py
(.eml -> parquet)    (28 indicators)  (4 models,     (figures)
                                       metrics)
predict.py  (classify a single .eml, explain which indicators fired)
ood_probe.py (out-of-distribution sanity check, see Limitations)
```

## Results (test set, n=2,390, 72.4% phishing)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Rules: any 2 indicator families | 0.916 | 0.232 | 0.370 | — |
| Rules: weighted score (thr=2.0) | 0.804 | 0.762 | 0.782 | 0.854 |
| LogReg: 28 indicators only | 0.994 | 0.967 | 0.980 | 0.998 |
| LogReg: TF-IDF only | 0.997 | 0.988 | 0.993 | 0.9995 |
| **LogReg: TF-IDF + indicators** | **0.997** | **0.985** | **0.991** | **0.9996** |

![PR curves](results/pr_curves.png)
![Threshold sweep](results/threshold_sweep.png)

### Precision/recall trade-offs

- **Rule-based scoring hits a wall.** Sweeping the rule threshold from 1 to 12
  trades recall from 0.93 down to 0.02 but precision never exceeds ~0.97, and
  anything above ~0.90 precision costs more than 80% of recall. Rules cannot
  be both precise and complete on this data.
- **The ML model's trade-off is nearly free.** Moving the probability
  threshold from 0.1 to 0.95 spans recall 0.997 → 0.936 while precision stays
  at 0.99+. At threshold 0.99 the model reaches perfect precision on this test
  set while still catching 68% of phish, a usable "auto-quarantine" operating
  point with a lower-threshold "flag for review" tier behind it.
- **Handcrafted indicators alone nearly match TF-IDF** (F1 0.980 vs 0.993).
  The interpretable features carry most of the signal; the 60k-feature text
  model adds ~1 point of F1.

### Why the rules baseline fails both ways

Rules misfire in both directions. Example from the prediction CLI: a legitimate
mailing-list email fires 7 rule points (reply-to differs, off-domain links,
brand mention) and gets flagged by rules, while the ML model scores it 0.018.
Meanwhile a Portuguese-language bank phish fires only 3 rule points (below
threshold 4) but the ML model scores it 0.991. Individual indicators are weak
evidence; the value is in learned weighting and interaction with text.

## Interpretability

- Strongest indicators by class separation: `has_html`, `links_offdomain`,
  `credential_terms`, `threat_terms`, `returnpath_differs`
  (see `results/indicator_prevalence.png`).
- `predict.py` outputs the fired indicators for any single email, so an
  analyst can see *why* something was flagged rather than trusting a bare score.

## Limitations (read before trusting the numbers)

1. **Temporal/domain confound.** The ham corpus is from 2002; the phishing
   corpus is 2022+. Some separation is corpus artifact, not phishing signal:
   top "legitimate" n-grams include mailing-list markers like "wrote", and top
   "phishing" n-grams include non-English stopwords because phishing_pot is
   multilingual while the ham is English-only. Mitigations applied: URLs,
   email addresses, and all digits are replaced with placeholder tokens before
   vectorization so the model cannot key on links or dates.
2. **OOD probe result.** To measure the confound, the trained model was run on
   SpamAssassin `spam_2`: 1,335 emails of 2002-era *malicious* mail (same era
   and formatting as the ham corpus, similar intent to phishing). The model
   flags **57% at threshold 0.5 (median score 0.88)**. If it had learned pure
   era artifacts, this number would be near zero. It transfers, but 57% on
   same-era malicious mail versus 98.5% in-distribution recall is the honest
   gap. Real generalization to current legitimate corporate email is untested
   because no public modern ham corpus with full headers exists.
3. **Prevalence is artificial.** 72% phishing prevalence reflects corpus
   sizes, not an inbox, where phishing is well under 1%. At realistic
   prevalence, even 0.997 precision here would translate to a much lower
   real-world precision. The threshold sweep and PR analysis are the right
   lens; the headline F1 is not deployable evidence.
4. **No adversarial robustness.** TF-IDF is trivially evadable (synonym swaps,
   image-based phish). The handcrafted URL/header indicators are harder to
   evade without giving up the attack's mechanics, which is why they matter.

## Repo layout

```
src/
  parse_emails.py   # .eml/raw -> structured parquet (headers, body, URLs, anchors)
  features.py       # 28 indicators: urgency, URL, sender-inconsistency
  rules.py          # rule-based baselines
  train.py          # 4-model benchmark, threshold sweeps, metrics.json
  make_plots.py     # figures
  predict.py        # classify one .eml + explain fired indicators
  ood_probe.py      # era-confound sanity check on spam_2
data/download_data.sh
results/            # metrics.json, ood_probe.json, 4 figures
models/             # saved vectorizer + scaler + model (joblib)
```

## Setup

```bash
pip install -r requirements.txt
bash data/download_data.sh
python src/parse_emails.py --phish data/raw/phishing_pot-main/email \
    --ham data/raw/spamassassin-public-corpus-main/easy_ham \
          data/raw/spamassassin-public-corpus-main/hard_ham \
    --out data/processed/emails.parquet
python src/features.py --data data/processed/emails.parquet --out data/processed/features.parquet
python src/train.py --data data/processed/emails.parquet --features data/processed/features.parquet
python src/make_plots.py
python src/predict.py path/to/suspicious.eml
```
