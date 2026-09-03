"""Configuration loading and experiment config dataclass."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, asdict
from pathlib import Path

from .constants import DEFAULT_MODEL_NAME


@dataclass
class ExperimentConfig:
    """Full experiment configuration loaded from YAML."""

    # Experiment identity
    config_id: str = "C00"
    experiment_name: str = "baseline"

    # Compression settings
    transform_type: str = "none"      # "dct", "fft", "none"
    filter_type: str = "none"         # "fixed", "learnable", "none"
    gamma: float = 1.0                # Compression ratio
    max_seq_len: int = 16384
    init_sharpness: float = 10.0
    init_offset: float = 0.0

    # Model settings
    model_name: str = DEFAULT_MODEL_NAME
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training settings
    redpajama_steps: int = 1000
    redpajama_batch_size: int = 64
    longalpaca_epochs: int = 5
    longalpaca_context_length: int = 16384
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Evaluation settings
    num_seeds: int = 30
    pg19_samples: int = 100
    proof_pile_samples: int = 100
    longbench_tasks: list[str] | None = None
    eval_window_size: int = 256
    eval_temperature: float = 0.7
    eval_top_p: float = 0.9

    # Infrastructure
    num_gpus: int = 8
    batch_size_per_gpu: int = 8
    deepspeed_config: str | None = None
    output_dir: str = "results"
    wandb_project: str = "csce823-spectral-kv"

    # Random seeds for evaluation
    seeds: list[int] | None = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(range(self.num_seeds))

    @property
    def variant_name(self) -> str:
        if self.transform_type == "none":
            return "baseline"
        return f"{self.transform_type}_{self.filter_type}"

    @property
    def is_baseline(self) -> bool:
        return self.gamma >= 1.0 or self.transform_type == "none"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        ExperimentConfig populated from the file.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Save experiment configuration to a YAML file.

    Args:
        config: ExperimentConfig to save.
        path: Output file path.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
    return path
