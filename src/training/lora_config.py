"""LoRA configuration for fine-tuning Llama-3.1-8B-Instruct.

Following FreqKV's protocol: LoRA rank 8, targeting attention projection layers.
The learnable spectral filter parameters are preserved via modules_to_save
so they are not frozen by LoRA and can be trained alongside the adapters.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    modules_to_save: list[str] | None = None,
) -> LoraConfig:
    """Create LoRA configuration for Llama-3.1-8B-Instruct.

    LoRA adapters are applied to attention projections (q, k, v, o).
    The learnable spectral filter parameters are saved as full-precision
    modules (not LoRA-adapted) so they train normally.

    Args:
        rank: LoRA rank. Default 8 per FreqKV protocol.
        alpha: LoRA alpha (scaling = alpha / rank). Default 16.
        dropout: LoRA dropout rate. Default 0.05.
        target_modules: Modules to apply LoRA to. Defaults to attention projections.
        modules_to_save: Full modules to save (not LoRA-adapted).
                        Defaults to spectral_cache.filter for learnable filters.

    Returns:
        peft.LoraConfig ready for use with get_peft_model().
    """
    if target_modules is None:
        # Llama-3 attention projections
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if modules_to_save is None:
        # Save the learnable spectral filter parameters
        # These are nested as: self_attn.spectral_cache.filter
        # PEFT's modules_to_save matches by module name suffix
        modules_to_save = ["spectral_cache"]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        modules_to_save=modules_to_save,
    )
