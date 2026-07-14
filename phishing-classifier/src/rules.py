"""
rules.py
Rule-based baselines, the kind a SOC might deploy as a first-pass filter.
These are the benchmark the ML model has to beat.

Two baselines:
  RuleScorer  - weighted indicator score with a tunable threshold, which lets
                us sweep thresholds and draw a PR curve for a rules approach.
  AnyTwoRules - binary "flag if >= k indicator families fire" heuristic
                (no score, single operating point).
"""

import numpy as np
import pandas as pd

# Weights are hand-set from security intuition, not fit to data. That is the
# point: this is what you can build without labels. Each tuple is
# (feature, weight, cap) where cap limits how much a repeated indicator adds.
RULE_WEIGHTS = [
    # urgency / social engineering
    ("urgency_terms",              1.0, 3),
    ("threat_terms",               1.5, 3),
    ("credential_terms",           1.5, 3),
    ("reward_terms",               1.0, 3),
    ("subject_all_caps_words",     0.5, 2),
    # suspicious URLs
    ("url_ip_host",                3.0, 1),
    ("url_shortener",              1.5, 1),
    ("url_at_symbol",              2.0, 1),
    ("url_suspicious_tld",         2.0, 2),
    ("url_punycode",               3.0, 1),
    ("anchor_href_mismatch",       2.0, 2),
    ("url_many_subdomains",        1.0, 2),
    # sender inconsistencies
    ("replyto_differs",            2.0, 1),
    ("brand_name_freemail",        3.0, 1),
    ("brand_domain_mismatch",      3.0, 1),
    ("display_name_addr_mismatch", 3.0, 1),
    ("links_offdomain",            1.0, 1),
]

FAMILIES = {
    "urgency": ["urgency_terms", "threat_terms", "credential_terms",
                "reward_terms"],
    "url": ["url_ip_host", "url_shortener", "url_at_symbol",
            "url_suspicious_tld", "url_punycode", "anchor_href_mismatch"],
    "sender": ["replyto_differs", "brand_name_freemail",
               "brand_domain_mismatch", "display_name_addr_mismatch"],
}


class RuleScorer:
    """Weighted indicator score. score() returns a continuous value so we can
    trade precision against recall by moving the threshold."""

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        s = np.zeros(len(feats))
        for col, w, cap in RULE_WEIGHTS:
            if col in feats:
                s += w * np.minimum(feats[col].values, cap)
        return s

    def predict(self, feats: pd.DataFrame, threshold: float = 4.0) -> np.ndarray:
        return (self.score(feats) >= threshold).astype(int)


class AnyTwoRules:
    """Flag an email if at least `k` distinct indicator families fire."""

    def __init__(self, k: int = 2):
        self.k = k

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        fired = np.zeros(len(feats))
        for cols in FAMILIES.values():
            cols = [c for c in cols if c in feats]
            fired += (feats[cols].sum(axis=1) > 0).astype(int).values
        return (fired >= self.k).astype(int)
