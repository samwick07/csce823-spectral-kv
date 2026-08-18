"""LoRA configuration for fine-tuning Llama-3-8B-Instruct.

Following FreqKV's protocol: LoRA rank 8, targeting attention projection layers.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Create LoRA configuration for Llama-3-8B-Instruct.

    Args:
        rank: LoRA rank. Default 8 per FreqKV protocol.
        alpha: LoRA alpha (scaling = alpha / rank). Default 16.
        dropout: LoRA dropout rate. Default 0.05.
        target_modules: Modules to apply LoRA to. Defaults to attention projections.

    Returns:
        peft.LoraConfig ready for use with get_peft_model().
    """
    if target_modules is None:
        # Llama-3 attention projections
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
