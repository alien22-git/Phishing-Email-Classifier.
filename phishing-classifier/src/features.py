"""
features.py
Handcrafted phishing indicators, grouped into the three families from the
project spec:

  1. Urgency / social-engineering language
  2. Suspicious URL characteristics
  3. Sender inconsistencies (header analysis)

Each function returns interpretable numeric features. These are used both by
the rule-based baseline (rules.py) and as an optional feature block appended
to TF-IDF for the ML model.
"""

import re
from email.utils import parseaddr
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Urgency / social-engineering language
# ---------------------------------------------------------------------------

URGENCY_TERMS = [
    "urgent", "immediately", "act now", "action required", "final notice",
    "last chance", "expire", "expires", "expired", "deadline", "asap",
    "within 24 hours", "within 48 hours", "right away", "attention",
    "important notice", "time sensitive", "don't delay", "hurry",
]

THREAT_TERMS = [
    "suspended", "suspension", "locked", "disabled", "deactivated",
    "unauthorized", "unusual activity", "suspicious activity",
    "security alert", "compromised", "terminated", "closed permanently",
    "failure to", "will be deleted", "on hold", "restricted",
]

CREDENTIAL_TERMS = [
    "verify your account", "verify your identity", "confirm your account",
    "confirm your identity", "update your information", "update your account",
    "update your payment", "validate your", "re-enter your", "login to",
    "log in to", "sign in to", "click here", "click the link", "click below",
    "password", "credentials", "billing information", "payment information",
    "social security", "ssn", "credit card",
]

REWARD_TERMS = [
    "you have won", "winner", "congratulations", "claim your", "prize",
    "reward", "gift card", "refund", "free", "bonus", "lottery",
    "inheritance", "beneficiary", "million dollars",
]


def _count_terms(text, terms):
    text = text.lower()
    return sum(text.count(t) for t in terms)


def urgency_features(subject, body):
    text = f"{subject} {body}"
    subj = subject or ""
    n_words = max(len(text.split()), 1)
    exclam = text.count("!")
    caps_words = sum(1 for w in subj.split() if len(w) > 2 and w.isupper())
    return {
        "urgency_terms": _count_terms(text, URGENCY_TERMS),
        "threat_terms": _count_terms(text, THREAT_TERMS),
        "credential_terms": _count_terms(text, CREDENTIAL_TERMS),
        "reward_terms": _count_terms(text, REWARD_TERMS),
        "exclamations_per_100w": 100.0 * exclam / n_words,
        "subject_all_caps_words": caps_words,
        "subject_has_re_fwd_fake": int(bool(re.match(r"\s*(re|fwd?):", subj.lower()))),
    }


# ---------------------------------------------------------------------------
# 2. Suspicious URL characteristics
# ---------------------------------------------------------------------------

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "rebrand.ly", "tiny.cc", "lnkd.in",
    "s.id", "qrco.de",
}

SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "click", "link", "live", "icu", "cam",
    "rest", "gq", "cf", "tk", "ml", "buzz", "monster", "quest", "cyou",
}

IP_HOST_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _host(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _registered_domain(host):
    """Crude eTLD+1: last two labels (good enough without a PSL dependency)."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def url_features(urls, anchor_pairs, body):
    feats = {
        "n_urls": len(urls),
        "url_ip_host": 0,
        "url_shortener": 0,
        "url_at_symbol": 0,
        "url_many_subdomains": 0,
        "url_suspicious_tld": 0,
        "url_punycode": 0,
        "url_long": 0,
        "url_port": 0,
        "url_hex_chars": 0,
        "anchor_href_mismatch": 0,
        "n_distinct_domains": 0,
    }
    domains = set()
    for u in urls:
        h = _host(u)
        if not h:
            continue
        domains.add(_registered_domain(h))
        if IP_HOST_RE.match(h):
            feats["url_ip_host"] += 1
        if _registered_domain(h) in SHORTENERS:
            feats["url_shortener"] += 1
        if "@" in u.split("://", 1)[-1].split("/")[0]:
            feats["url_at_symbol"] += 1
        if h.count(".") >= 3:
            feats["url_many_subdomains"] += 1
        if h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
            feats["url_suspicious_tld"] += 1
        if "xn--" in h:
            feats["url_punycode"] += 1
        if len(u) > 120:
            feats["url_long"] += 1
        if re.search(r":\d{2,5}(/|$)", u.split(h, 1)[-1][:6] if h in u else ""):
            feats["url_port"] += 1
        if len(re.findall(r"%[0-9a-fA-F]{2}", u)) >= 4:
            feats["url_hex_chars"] += 1

    # Anchor text says one domain, href points to another (classic phish trick)
    for anchor_text, href in anchor_pairs:
        href_host = _registered_domain(_host(href))
        m = URL_IN_TEXT_RE.search(anchor_text or "")
        if m:
            text_host = _registered_domain(_host(m.group(0)))
            if text_host and href_host and text_host != href_host:
                feats["anchor_href_mismatch"] += 1

    feats["n_distinct_domains"] = len(domains)
    return feats


URL_IN_TEXT_RE = re.compile(r"""https?://[^\s<>"']+|www\.[^\s<>"']+""", re.I)


# ---------------------------------------------------------------------------
# 3. Sender inconsistencies
# ---------------------------------------------------------------------------

FREEMAIL = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "mail.com", "gmx.com", "gmx.net", "protonmail.com", "proton.me",
    "icloud.com", "yandex.com", "zoho.com", "live.com", "msn.com",
}

BRAND_TERMS = [
    "paypal", "amazon", "apple", "microsoft", "netflix", "facebook", "meta",
    "instagram", "google", "chase", "wells fargo", "bank of america",
    "docusign", "dhl", "fedex", "usps", "ups", "irs", "geek squad",
    "norton", "mcafee", "coinbase", "binance", "att", "verizon", "adobe",
]


def _domain_of(addr_header):
    _, addr = parseaddr(addr_header or "")
    if "@" in addr:
        return addr.rsplit("@", 1)[-1].lower().strip(">").strip()
    return ""


def sender_features(from_raw, reply_to, return_path, urls):
    disp_name, from_addr = parseaddr(from_raw or "")
    from_dom = _domain_of(from_raw)
    reply_dom = _domain_of(reply_to)
    rpath_dom = _domain_of(return_path)

    from_reg = _registered_domain(from_dom) if from_dom else ""
    reply_reg = _registered_domain(reply_dom) if reply_dom else ""
    rpath_reg = _registered_domain(rpath_dom) if rpath_dom else ""

    name_l = (disp_name or "").lower()
    brand_in_name = next((b for b in BRAND_TERMS if b in name_l), None)

    # Display name contains a brand, but the sending domain is unrelated
    brand_domain_mismatch = 0
    if brand_in_name and from_reg:
        if brand_in_name.replace(" ", "") not in from_reg.replace("-", ""):
            brand_domain_mismatch = 1

    # Display name itself looks like an email address from a different domain
    name_addr_mismatch = 0
    m = re.search(r"[\w.+-]+@([\w-]+\.[\w.-]+)", disp_name or "")
    if m and from_reg:
        if _registered_domain(m.group(1).lower()) != from_reg:
            name_addr_mismatch = 1

    link_doms = {_registered_domain(_host(u)) for u in urls if _host(u)}

    return {
        "replyto_differs": int(bool(reply_reg) and reply_reg != from_reg),
        "returnpath_differs": int(bool(rpath_reg) and bool(from_reg)
                                  and rpath_reg != from_reg),
        "brand_name_freemail": int(bool(brand_in_name) and from_reg in FREEMAIL),
        "brand_domain_mismatch": brand_domain_mismatch,
        "display_name_addr_mismatch": name_addr_mismatch,
        "from_missing": int(not from_addr),
        "links_offdomain": int(bool(from_reg) and bool(link_doms)
                               and from_reg not in link_doms),
    }


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        f = {}
        f.update(urgency_features(r.subject or "", r.body or ""))
        f.update(url_features(list(r.urls), list(r.anchor_pairs), r.body or ""))
        f.update(sender_features(r.from_raw, r.reply_to, r.return_path,
                                 list(r.urls)))
        f["has_html"] = int(r.has_html)
        f["n_attachments"] = r.n_attachments
        rows.append(f)
    out = pd.DataFrame(rows, index=df.index).astype(float)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.data)
    feats = build_feature_frame(df)
    feats["label"] = df["label"].values
    feats.to_parquet(args.out)
    print(f"Built {feats.shape[1] - 1} features for {len(feats)} emails -> {args.out}")
