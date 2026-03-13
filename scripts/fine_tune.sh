#!/usr/bin/env bash
set -euo pipefail

echo "=== Nemotron-Nano-9B LoRA Fine-Tuning ==="
echo ""

EPOCHS=${1:-3}
LR=${2:-2e-4}
BATCH=${3:-4}

echo "Config: epochs=$EPOCHS, lr=$LR, batch=$BATCH"
echo "Training data: backend/training_data/pricing_qa.jsonl"
echo ""

cd backend

# Ensure fine-tuning dependencies are installed
../.venv/bin/pip install peft accelerate datasets bitsandbytes 2>/dev/null || true

# Run fine-tuning
../.venv/bin/python -m services.fine_tune \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH"

echo ""
echo "=== Fine-tuning complete ==="
echo "LoRA adapter saved to: backend/training_data/lora_output/final/"
echo ""
echo "To push to HuggingFace Hub, run:"
echo "  cd backend && ../.venv/bin/python -m services.fine_tune --push-to-hub"
