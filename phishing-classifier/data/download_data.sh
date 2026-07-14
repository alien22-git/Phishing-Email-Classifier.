#!/usr/bin/env bash
# Downloads both corpora into data/raw/
set -e
mkdir -p data/raw && cd data/raw
echo "[1/2] phishing_pot (real phishing samples, ~210MB)..."
curl -L -o phishing_pot.tar.gz "https://codeload.github.com/rf-peixoto/phishing_pot/tar.gz/refs/heads/main"
tar xzf phishing_pot.tar.gz
echo "[2/2] SpamAssassin public corpus mirror (~6MB)..."
curl -L -o spamassassin.tar.gz "https://codeload.github.com/reyhanmufiid/spamassassin-public-corpus/tar.gz/refs/heads/main"
tar xzf spamassassin.tar.gz
echo "Done. Raw data in data/raw/"
