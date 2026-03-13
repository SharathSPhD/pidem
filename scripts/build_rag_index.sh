#!/usr/bin/env bash
set -euo pipefail

echo "=== Building RAG Index ==="
echo ""

cd backend

# Build the FAISS index from the embedded corpus
../.venv/bin/python -c "
from services.rag_pipeline import build_index
n = build_index(force=True)
print(f'Index built with {n} chunks')
"

echo ""
echo "=== RAG index ready ==="
echo "The index is held in memory. It will be rebuilt on server restart."
echo "To add documents, place .md or .txt files in backend/data/corpus/"
