"""
parse_emails.py
Parse raw email files (.eml / SpamAssassin format) into a structured dataset.

Extracts: subject, body text, From / Reply-To / Return-Path headers,
display name, and all URLs (both raw and anchor-text/href pairs from HTML).

Usage:
    python src/parse_emails.py --phish data/raw/phishing_pot-main/email \
        --ham data/raw/spamassassin-public-corpus-main/easy_ham \
              data/raw/spamassassin-public-corpus-main/hard_ham \
        --out data/processed/emails.parquet
"""

import argparse
import email
import email.policy
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

URL_RE = re.compile(r"""https?://[^\s<>"'\)\]]+""", re.IGNORECASE)


def extract_body_and_urls(msg):
    """Return (plain_text, list_of_urls, list_of_(anchor_text, href)) from a message."""
    text_parts, urls, anchors = [], [], []
    html_seen = False

    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_content()
        except Exception:
            try:
                payload = part.get_payload(decode=True)
                if payload is not None:
                    payload = payload.decode("utf-8", errors="replace")
            except Exception:
                continue
        if not isinstance(payload, str):
            continue

        if ctype == "text/html":
            html_seen = True
            soup = BeautifulSoup(payload, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                anchor_text = a.get_text(" ", strip=True)
                anchors.append((anchor_text, href))
                if href.lower().startswith("http"):
                    urls.append(href)
            text = soup.get_text(" ", strip=True)
        else:
            text = payload

        text_parts.append(text)
        urls.extend(URL_RE.findall(text))

    body = "\n".join(text_parts)
    # dedupe URLs, keep order
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return body, uniq, anchors, html_seen


def parse_file(path: Path, label: int, source: str):
    raw = path.read_bytes()
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    except Exception:
        return None

    try:
        body, urls, anchors, html_seen = extract_body_and_urls(msg)
    except Exception:
        return None

    def hdr(name):
        try:
            v = msg.get(name)
            return str(v) if v is not None else ""
        except Exception:
            return ""

    return {
        "id": path.name,
        "source": source,
        "label": label,  # 1 = phishing, 0 = legitimate
        "subject": hdr("Subject"),
        "from_raw": hdr("From"),
        "reply_to": hdr("Reply-To"),
        "return_path": hdr("Return-Path"),
        "to_raw": hdr("To"),
        "body": body,
        "urls": urls,
        "anchor_pairs": anchors,
        "has_html": html_seen,
        "n_attachments": sum(
            1 for p in msg.walk()
            if p.get_content_disposition() == "attachment"
        ),
    }


def collect(dirs, label, source_prefix):
    rows = []
    for d in dirs:
        d = Path(d)
        files = [f for f in d.rglob("*") if f.is_file() and "MACOSX" not in str(f)]
        for f in files:
            row = parse_file(f, label, f"{source_prefix}:{d.name}")
            if row is not None and (row["body"].strip() or row["subject"].strip()):
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phish", nargs="+", required=True)
    ap.add_argument("--ham", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = collect(args.phish, 1, "phish") + collect(args.ham, 0, "ham")
    df = pd.DataFrame(rows)

    # Drop exact duplicates on subject+body (phishing kits resend identical mail;
    # duplicates across a random split leak train data into the test set)
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    print(f"Parsed {before} emails, {len(df)} after dedup "
          f"({df['label'].sum()} phishing / {(df['label'] == 0).sum()} ham)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
