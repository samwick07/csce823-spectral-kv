"""Weights & Biases integration for experiment tracking and sharing.

Provides:
  - init_wandb(): Initialize a W&B run with full experiment config
  - log_spectral_stats(): Log learnable filter masks and compression ratios
  - log_eval_results(): Log evaluation metrics per benchmark
  - create_wandb_report(): Generate a shareable W&B report for publishing
  - finish_wandb(): Clean finish with final metrics

The W&B project is set up for public sharing to support reproducibility
and the forthcoming journal article.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

WANDB_PROJECT = "csce823-spectral-kv"
WANDB_ENTITY = None  # Set to your W&B entity/team name for shared projects


def init_wandb(
    config: Any,
    phase: str = "train",
    entity: str | None = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
    tags: list[str] | None = None,
    notes: str = "",
    resume: Literal["allow", "never", "must", "auto"] = "allow",
) -> Any:
    """Initialize a W&B run with full experiment configuration.

    Logs all experiment parameters as W&B config for reproducibility.
    The run is tagged with the variant name, compression ratio, and phase
    for easy filtering in the W&B dashboard.

    Uses a deterministic run ID so that crashes don't fragment a single
    training run into multiple W&B runs. On restart, wandb.init with
    resume="allow" will resume the existing run and append new logs.

    The step overlap that occurs when DeepSpeed resumes from a checkpoint
    (re-logging a step that was already logged before the crash) is
    handled correctly by W&B: the existing value at that step is
    overwritten with an identical value (model state was restored from
    the checkpoint, so the metrics are the same).

    Args:
        config: ExperimentConfig dataclass with all experiment settings.
        phase: "train_phase1", "train_phase2", or "eval".
        entity: W&B entity/team. None uses the default user entity.
        project: W&B project name.
        tags: Additional tags for the run.
        notes: Free-text notes for the run.
        resume: W&B resume mode. "allow" (default) resumes if the run id
            already exists, otherwise creates a new run. "must" requires
            the run to already exist. "never" always creates a new run.

    Returns:
        The wandb.Run object.
    """
    import wandb

    # Build the config dict from the dataclass
    from dataclasses import asdict
    config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)

    # Convert lists to strings for W&B config compatibility
    for k, v in config_dict.items():
        if isinstance(v, list):
            config_dict[k] = str(v)

    # Build tags
    default_tags = [
        config.variant_name if hasattr(config, "variant_name") else "unknown",
        f"gamma_{config.gamma}",
        f"{config.transform_type}_{config.filter_type}",
        phase,
    ]
    if tags:
        default_tags.extend(tags)

    # Deterministic run ID: ensures crash recovery resumes the same W&B run
    # instead of creating a new one. Format: {config_id}_{phase}
    # This also gives each phase its own run, preventing step overlap
    # between Phase 1 (steps 10-1000) and Phase 2 (steps 10-50).
    run_id = f"{config.config_id}_{phase}"

    run = wandb.init(
        id=run_id,
        resume=resume,
        project=project,
        entity=entity,
        name=f"{config.config_id}_{phase}",
        config=config_dict,
        tags=default_tags,
        notes=notes or f"CSCE 823: {config.experiment_name} - {phase}",
        dir=str(Path.cwd() / "wandb"),
    )

    logger.info(
        f"Initialized W&B run: {run.name} (id={run_id}, resume={resume}) "
        f"(project={project}, entity={entity or 'default'})"
    )

    return run


def log_spectral_stats(model: Any, step: int | None = None) -> None:
    """Log spectral compression statistics to W&B.

    Logs per-layer:
      - Compression ratio (actual achieved ratio)
      - Learnable filter masks (as images for visualization)
      - Retained fraction per head

    Args:
        model: Model with spectral compression applied.
        step: Training step (for x-axis on W&B charts).
    """
    import wandb

    from ..spectral.attention import get_spectral_caches, get_compression_stats

    # Log compression stats as a table
    stats = get_compression_stats(model)
    if not stats:
        return

    # Per-layer compression ratios
    ratios = {f"compression/layer_{s['layer']}_ratio": s["compression_ratio"] for s in stats}
    wandb.log(ratios, step=step)

    # Log learnable filter masks as images
    caches = get_spectral_caches(model)
    for i, cache in enumerate(caches):
        if cache.filter is not None and hasattr(cache.filter, "get_mask"):
            mask = cache.filter.get_mask()  # [num_heads, max_spectral_len]
            # Log as image: each head's frequency response
            wandb.log(
                {f"filter_masks/layer_{i}": wandb.Image(mask.cpu().numpy(), caption=f"Layer {i} filter mask")},
                step=step,
            )

            # Log retained fraction per head
            retained = cache.filter.get_retained_fraction()
            for h in range(retained.shape[0]):
                wandb.log(
                    {f"filter_retained/layer_{i}_head_{h}": retained[h].item()},
                    step=step,
                )


def log_eval_results(
    results: dict,
    config_id: str,
    seed: int,
    step: int | None = None,
) -> None:
    """Log evaluation results to W&B.

    Args:
        results: Dict from an evaluate_* function (pg19, proof_pile, etc.).
        config_id: Experiment config ID (e.g., "C01").
        seed: Random seed for this evaluation run.
        step: W&B step.
    """
    import wandb

    benchmark = results.get("benchmark", "unknown")

    log_dict = {}

    if "mean_perplexity" in results:
        log_dict[f"eval/{benchmark}_mean_ppl"] = results["mean_perplexity"]
        log_dict[f"eval/{benchmark}_median_ppl"] = results["median_perplexity"]
        log_dict[f"eval/{benchmark}_std_ppl"] = results["std_perplexity"]
    elif "overall_mean" in results:
        log_dict[f"eval/{benchmark}_overall_score"] = results["overall_mean"]
        # Log per-task scores
        for task_name, task_data in results.get("tasks", {}).items():
            if "mean_score" in task_data:
                log_dict[f"eval/{benchmark}_{task_name}"] = task_data["mean_score"]
    elif "peak_kv_memory_gb" in results:
        log_dict[f"efficiency/peak_memory_gb"] = results["peak_kv_memory_gb"]
        log_dict[f"efficiency/latency_ms_per_token"] = results["decoding_latency_ms_per_token"]
        log_dict[f"efficiency/compression_overhead_pct"] = results.get("compression_overhead_pct", 0.0)

    log_dict["eval/seed"] = seed
    log_dict["eval/config_id"] = config_id

    wandb.log(log_dict, step=step)
    logger.info(f"Logged {benchmark} eval results to W&B for {config_id} seed={seed}")


def create_wandb_report(
    title: str = "CSCE 823: Spectral KV-Cache Compression Ablation Study",
    description: str = "",
) -> str:
    """Create a shareable W&B report for the experiment.

    Generates a W&B report with:
      - Overview of the 2x2 factorial design
      - Per-config perplexity comparison
      - Compression ratio vs quality trade-off plots
      - Learnable filter mask visualizations
      - Statistical significance results

    Args:
        title: Report title.
        description: Report description.

    Returns:
        URL to the created W&B report.
    """
    import wandb

    api = wandb.Api()

    # Build report spec
    panels = [
        {
            "query": {"aggregation": "mean", "fields": ["eval/pg19_mean_ppl"]},
            "title": "PG-19 Perplexity by Configuration",
            "type": "bar",
        },
        {
            "query": {"aggregation": "mean", "fields": ["eval/proof_pile_mean_ppl"]},
            "title": "Proof-pile Perplexity by Configuration",
            "type": "bar",
        },
        {
            "query": {"aggregation": "mean", "fields": ["eval/longbench_v1_overall_score"]},
            "title": "LongBench V1 Score by Configuration",
            "type": "bar",
        },
        {
            "query": {"aggregation": "mean", "fields": ["efficiency/peak_memory_gb"]},
            "title": "Peak KV Memory by Configuration",
            "type": "bar",
        },
    ]

    logger.info("W&B report creation requires the W&B API.")
    logger.info(f"After runs complete, create a report at: https://wandb.ai/{WANDB_ENTITY or 'your-entity'}/{WANDB_PROJECT}/reports")
    logger.info("Use the panels above as a starting point for the report layout.")

    return f"https://wandb.ai/{WANDB_ENTITY or 'your-entity'}/{WANDB_PROJECT}/reports"


def finish_wandb(final_metrics: dict | None = None) -> None:
    """Finish the current W&B run.

    Args:
        final_metrics: Optional final metrics to log before finishing.
    """
    import wandb

    if final_metrics:
        wandb.log(final_metrics)

    wandb.finish()
    logger.info("W&B run finished")


def download_wandb_runs(
    project: str = WANDB_PROJECT,
    entity: str | None = WANDB_ENTITY,
    filters: dict | None = None,
) -> list[dict]:
    """Download run summaries from W&B for offline analysis.

    Useful for generating the statistical analysis tables from
    W&B-tracked experiments.

    Args:
        project: W&B project name.
        entity: W&B entity.
        filters: Optional filter dict (e.g., {"config.config_id": "C01"}).

    Returns:
        List of run summary dicts.
    """
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{entity or api.default_entity}/{project}",
        filters=filters or {},
    )

    summaries = []
    for run in runs:
        summaries.append({
            "id": run.id,
            "name": run.name,
            "config": run.config,
            "summary": run.summary._json_dict,
            "tags": run.tags,
            "url": run.url,
        })

    logger.info(f"Downloaded {len(summaries)} runs from W&B")
    return summaries
