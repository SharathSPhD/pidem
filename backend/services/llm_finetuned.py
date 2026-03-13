"""
Fine-tuned Nemotron inference endpoint.

Loads the base model + LoRA adapter and serves curriculum-specific tasks
(elasticity interpretation, SHAP explanation, strategy recommendation, etc.)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

LORA_PATH = Path(__file__).parent.parent / "training_data" / "lora_output" / "final"
BASE_MODEL = "nvidia/Nemotron-Nano-9B-v2"


def _load_finetuned():
    """Load base model with merged LoRA weights."""
    global _model, _tokenizer

    if _model is not None:
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        if LORA_PATH.exists():
            logger.info(f"Loading fine-tuned model from {LORA_PATH}")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
            _model = PeftModel.from_pretrained(base, str(LORA_PATH))
            _model = _model.merge_and_unload()
            logger.info("LoRA adapter merged successfully")
        else:
            logger.warning(f"LoRA adapter not found at {LORA_PATH}, using base model")
            _model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
    except ImportError as e:
        logger.error(f"Cannot load fine-tuned model: {e}")
        _model = "unavailable"
        _tokenizer = "unavailable"


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
    """Generate a response using the fine-tuned model."""
    _load_finetuned()

    if _model == "unavailable" or _model is None:
        return (
            f"[Fine-tuned model not available]\n\n"
            f"To use the fine-tuned endpoint, run the LoRA training first:\n"
            f"  cd backend && python -m services.fine_tune\n\n"
            f"Your query: {prompt[:200]}"
        )

    try:
        import torch

        formatted = (
            f"<|system|>\nYou are an expert pricing analyst.\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )

        inputs = _tokenizer(formatted, return_tensors="pt").to(_model.device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )

        response = _tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as exc:
        logger.warning("Fine-tuned generation failed: %s", exc)
        return (
            "[Fine-tuned generation unavailable in this environment]\n\n"
            f"Reason: {exc}\n"
            "Run on a GPU-capable environment with transformers/peft weights installed."
        )


CURRICULUM_TASKS = {
    "elasticity": "Interpret these elasticity coefficients in business terms",
    "shap": "Explain these SHAP values for a pricing analyst",
    "diagnostic": "Interpret these model diagnostics and suggest improvements",
    "strategy": "Recommend a pricing strategy based on this analysis",
    "competitor": "Assess this competitive situation and recommend a response",
}


def curriculum_generate(task: str, context: str, **kwargs) -> str:
    """Generate a curriculum-specific response."""
    task_prompt = CURRICULUM_TASKS.get(task, "Analyze the following:")
    full_prompt = f"{task_prompt}:\n\n{context}"
    return generate(full_prompt, **kwargs)
