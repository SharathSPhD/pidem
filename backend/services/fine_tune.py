"""
LoRA fine-tuning script for Nemotron-Nano-9B on curriculum-specific pricing Q&A.

Usage (standalone):
  python -m services.fine_tune --epochs 3 --lr 2e-4

Designed for DGX Spark GB10 (128GB unified memory).
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TRAINING_DATA_PATH = Path(__file__).parent.parent / "training_data" / "pricing_qa.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "training_data" / "lora_output"
BASE_MODEL = "nvidia/Nemotron-Nano-9B-v2"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_training_data() -> list[dict]:
    """Load pricing Q&A examples from JSONL."""
    if not TRAINING_DATA_PATH.exists():
        logger.warning(f"Training data not found at {TRAINING_DATA_PATH}")
        return []

    examples = []
    with open(TRAINING_DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info(f"Loaded {len(examples)} training examples")
    return examples


def format_for_training(examples: list[dict]) -> list[dict]:
    """Format examples into instruction-following format."""
    formatted = []
    for ex in examples:
        text = (
            f"<|system|>\nYou are an expert pricing analyst.\n"
            f"<|user|>\n{ex.get('question', ex.get('input', ''))}\n"
            f"<|assistant|>\n{ex.get('answer', ex.get('output', ''))}"
        )
        formatted.append({"text": text})
    return formatted


def run_fine_tuning(
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    use_wandb: bool = True,
    push_to_hub: bool = False,
    hub_model_id: str = "SharathSPhD/nemotron-pricing-lora",
):
    """Run LoRA fine-tuning on the base Nemotron model."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing dependency for fine-tuning: {e}")
        logger.info("Install with: pip install transformers peft datasets accelerate")
        return {"status": "error", "message": str(e)}

    examples = load_training_data()
    if len(examples) < 10:
        return {"status": "error", "message": f"Need at least 10 examples, found {len(examples)}"}

    formatted = format_for_training(examples)
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )

    tokenized_train = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    tokenized_eval = split["test"].map(tokenize, batched=True, remove_columns=["text"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="wandb" if use_wandb else "none",
        run_name="nemotron-pricing-lora",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    logger.info("Starting LoRA fine-tuning...")
    result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))

    if push_to_hub:
        try:
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            logger.info(f"Pushed LoRA adapter to {hub_model_id}")
        except Exception as e:
            logger.warning(f"Failed to push to Hub: {e}")

    return {
        "status": "complete",
        "train_loss": result.training_loss,
        "epochs": epochs,
        "output_dir": str(OUTPUT_DIR / "final"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron for pricing")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_fine_tuning(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
        push_to_hub=args.push_to_hub,
    )
    print(json.dumps(result, indent=2))
