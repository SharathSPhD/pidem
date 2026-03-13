#!/usr/bin/env bash
# Add, commit, and push to GitHub in one step. Uses token from .env (see push_with_token.sh).
# Usage: ./scripts/git-acp.sh "Your commit message"
#        ./scripts/git-acp.sh   (uses message "Update")

set -e
cd "$(dirname "$0")/.."

MSG="${1:-Update}"

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

echo "→ git add -A"
git add -A

echo "→ git status"
git status --short

if [[ -z $(git status -s) ]]; then
  echo "Nothing to commit. Exiting."
  exit 0
fi

echo "→ git commit -m \"$MSG\""
git commit -m "$MSG"

echo "→ push to origin main"
git remote set-url origin "https://SharathSPhD:${github_token}@github.com/SharathSPhD/pidem.git"
git push -u origin main
git remote set-url origin "https://github.com/SharathSPhD/pidem.git"

echo "Done: add + commit + push."
