#!/usr/bin/env bash
# Push to GitHub using token from .env (github_token=...).
# Usage: ./scripts/push_with_token.sh
# Requires: .env in project root with github_token=ghp_...

set -e
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Missing .env. Add a line: github_token=YOUR_GITHUB_TOKEN"
  exit 1
fi

# shellcheck disable=SC1091
source .env

if [[ -z "${github_token}" ]]; then
  echo "Missing github_token in .env"
  exit 1
fi

export DEVELOPER_DIR=/Library/Developer/CommandLineTools

# Use token for this push only, then remove it from remote URL
git remote set-url origin "https://SharathSPhD:${github_token}@github.com/SharathSPhD/pidem.git"
git push -u origin main --force
git remote set-url origin "https://github.com/SharathSPhD/pidem.git"

echo "Push done. Remote URL reset to HTTPS (no token stored)."
