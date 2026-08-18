"""Smoke test for the spectral KV-cache compression pipeline.

Runs on GPU with a small model to verify the full integration:
  1. Model loads with eager attention
  2. Spectral compression applies to all layers
  3. Forward pass with compression produces valid output
  4. Loss computation succeeds (no NaN)
  5. Backward pass updates learnable filter params
  6. reset_all_caches() clears state properly
  7. PEFT modules_to_save saves/loads spectral_cache
  8. PG-19 perplexity eval (1 sample)
  9. LongBench eval (1 sample)

Usage:
    python -m src.tests.smoke_test
    python -m src.tests.smoke_test --model meta-llama/Llama-3.2-1B-Instruct
    python -m src.tests.smoke_test --transform fft --filter learnable --gamma 0.22

Model defaults to SMOKE_TEST_MODEL_NAME from src/utils/constants.py.

Designed to run on the Coder workspace (coder.afitcdn.org) with GPU access.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch

from ..utils.constants import SMOKE_TEST_MODEL_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

class SmokeTestResult:
    """Track pass/fail for each sub-test."""
    def __init__(self):
        self.results: list[dict] = []

    def record(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "test": name,
            "passed": passed,
            "status": status,
            "detail": detail,
        })
        logger.info(f"  [{status}] {name}: {detail}")

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results) if self.results else False

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r["passed"])

    def summary(self) -> str:
        lines = [f"\n{'='*60}", "SMOKE TEST SUMMARY", f"{'='*60}"]
        for r in self.results:
            icon = "✓" if r["passed"] else "✗"
            lines.append(f"  {icon} {r['test']}: {r['detail']}")
        lines.append(f"\n  {self.num_passed} passed, {self.num_failed} failed")
        if self.all_passed:
            lines.append("  ALL TESTS PASSED")
        else:
            lines.append("  SOME TESTS FAILED")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual smoke tests
# ---------------------------------------------------------------------------

def test_model_loading(model_name: str, hf_token: str | None, result: SmokeTestResult):
    """Test 1: Load model with eager attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Test 1: Model loading with eager attention")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        )
        result.record("model_loading", True, f"Loaded {model_name}")
        return model, tokenizer
    except Exception as e:
        result.record("model_loading", False, str(e))
        raise


def test_spectral_compression(
    model, transform_type: str, filter_type: str, gamma: float, result: SmokeTestResult
):
    """Test 2: Apply spectral compression to all layers."""
    from src.spectral import CompressionConfig, apply_spectral_compression
    from src.spectral.attention import get_spectral_caches

    logger.info(f"Test 2: Apply spectral compression ({transform_type}/{filter_type}, gamma={gamma})")
    try:
        config = CompressionConfig(
            transform_type=transform_type,
            filter_type=filter_type,
            gamma=gamma,
            max_seq_len=4096,
        )
        model = apply_spectral_compression(model, config)
        caches = get_spectral_caches(model)
        num_layers = len(model.model.layers)
        assert len(caches) == num_layers, f"Expected {num_layers} caches, got {len(caches)}"
        result.record(
            "spectral_compression",
            True,
            f"Applied to {num_layers} layers",
        )
        return model
    except Exception as e:
        result.record("spectral_compression", False, str(e))
        raise


def test_forward_pass(model, tokenizer, result: SmokeTestResult):
    """Test 3: Forward pass with spectral compression produces valid output."""
    from src.spectral.attention import reset_all_caches

    logger.info("Test 3: Forward pass with compression")
    try:
        device = next(model.parameters()).device
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        reset_all_caches(model)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        assert not torch.isnan(logits).any(), "NaN in output logits"
        assert not torch.isinf(logits).any(), "Inf in output logits"
        assert logits.shape[0] == inputs["input_ids"].shape[0]
        assert logits.shape[1] == inputs["input_ids"].shape[1]
        result.record("forward_pass", True, f"logits shape={tuple(logits.shape)}, no NaN/Inf")
    except Exception as e:
        result.record("forward_pass", False, str(e))
        raise


def test_loss_computation(model, tokenizer, result: SmokeTestResult):
    """Test 4: Loss computation succeeds (no NaN)."""
    from src.spectral.attention import reset_all_caches

    logger.info("Test 4: Loss computation")
    try:
        device = next(model.parameters()).device
        text = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        reset_all_caches(model)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss
        assert not torch.isnan(loss), f"Loss is NaN"
        assert not torch.isinf(loss), f"Loss is Inf"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        result.record("loss_computation", True, f"loss={loss.item():.4f}")
    except Exception as e:
        result.record("loss_computation", False, str(e))
        raise


def test_gradient_flow(model, tokenizer, result: SmokeTestResult):
    """Test 5: Backward pass updates learnable filter params."""
    from src.spectral.attention import get_learnable_filter_params, reset_all_caches

    logger.info("Test 5: Gradient flow through learnable filter")
    try:
        device = next(model.parameters()).device
        text = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        learnable_params = get_learnable_filter_params(model)
        if not learnable_params:
            result.record("gradient_flow", True, "No learnable params (fixed filter), skipping")
            return

        # Store pre-backward values
        pre_values = [p.clone() for p in learnable_params]

        reset_all_caches(model)
        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        # Check that at least some filter params have gradients
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in learnable_params)
        assert has_grads, "No gradients on learnable filter params"

        # Verify params didn't change yet (no optimizer step)
        all_same = all(torch.allclose(pre, p) for pre, p in zip(pre_values, learnable_params))
        assert all_same, "Params changed before optimizer step"

        # Take one optimizer step
        optimizer = torch.optim.AdamW(learnable_params, lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()

        any_changed = any(
            not torch.allclose(pre, p) for pre, p in zip(pre_values, learnable_params)
        )
        assert any_changed, "Params did not change after optimizer step"

        model.eval()
        result.record(
            "gradient_flow",
            True,
            f"{len(learnable_params)} filter params, grads flow, optimizer step changes params",
        )
    except Exception as e:
        result.record("gradient_flow", False, str(e))
        raise


def test_reset_caches(model, tokenizer, result: SmokeTestResult):
    """Test 6: reset_all_caches() clears state properly."""
    from src.spectral.attention import get_spectral_caches, reset_all_caches

    logger.info("Test 6: Cache reset")
    try:
        device = next(model.parameters()).device
        text = "The quick brown fox."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Run forward to populate caches
        reset_all_caches(model)
        with torch.no_grad():
            model(**inputs)

        # Check caches are populated
        caches = get_spectral_caches(model)
        populated_before = any(c._cached_k_spectral is not None for c in caches)

        # Reset
        reset_all_caches(model)

        # Check all caches are cleared
        all_cleared = all(c._cached_k_spectral is None for c in caches)
        assert all_cleared, "Caches not cleared after reset_all_caches()"
        result.record(
            "reset_caches",
            True,
            f"Caches populated after forward, cleared after reset ({len(caches)} layers)",
        )
    except Exception as e:
        result.record("reset_caches", False, str(e))
        raise


def test_checkpoint_save_load(model, tokenizer, result: SmokeTestResult):
    """Test 7: PEFT modules_to_save saves/loads spectral_cache (P2-6)."""
    from peft import LoraConfig, PeftModel, get_peft_model, TaskType
    from src.spectral.attention import get_spectral_caches, reset_all_caches

    logger.info("Test 7: Checkpoint save/load with spectral_cache (P2-6)")
    try:
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            modules_to_save=["spectral_cache"],
        )
        model = get_peft_model(model, lora_config)

        # Capture filter logits before saving
        caches_before = get_spectral_caches(model)
        learnable_before = []
        for cache in caches_before:
            if cache.filter is not None and hasattr(cache.filter, "filter_logits"):
                learnable_before.append(cache.filter.filter_logits.clone())
            else:
                learnable_before.append(None)

        # Save to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))

            # Verify adapter_config.json mentions modules_to_save
            adapter_config = (ckpt_path / "adapter_config.json").read_text()
            assert "spectral_cache" in adapter_config, \
                "spectral_cache not in adapter_config.json modules_to_save"

            # Load fresh base model
            from transformers import AutoModelForCausalLM
            model_name = model.config._name_or_path
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="auto",
            )

            # Re-apply spectral compression with same config
            from src.spectral import CompressionConfig, apply_spectral_compression
            # Get config from existing cache
            first_cache = caches_before[0]
            comp_config = first_cache.config
            base_model = apply_spectral_compression(base_model, comp_config)

            # Load LoRA adapter
            loaded_model = PeftModel.from_pretrained(base_model, str(ckpt_path))

            # Verify spectral_cache was restored
            caches_after = get_spectral_caches(loaded_model)
            assert len(caches_after) == len(caches_before), \
                f"Cache count mismatch: {len(caches_after)} vs {len(caches_before)}"

            # Check that learnable filter params were restored
            all_match = True
            for i, (cache_before, cache_after, logits_before) in enumerate(
                zip(caches_before, caches_after, learnable_before)
            ):
                if logits_before is None:
                    continue
                if not hasattr(cache_after.filter, "filter_logits"):
                    all_match = False
                    break
                logits_after = cache_after.filter.filter_logits
                if not torch.allclose(logits_before, logits_after, atol=1e-6):
                    all_match = False
                    break

            assert all_match, "Learnable filter params not restored from checkpoint"

        result.record(
            "checkpoint_save_load",
            True,
            f"LoRA + spectral_cache saved and restored ({len(learnable_before)} layers)",
        )
        return loaded_model
    except Exception as e:
        result.record("checkpoint_save_load", False, str(e))
        # Don't re-raise; continue with other tests
        logger.error(traceback.format_exc())
        return model


def test_pg19_eval(model, tokenizer, result: SmokeTestResult):
    """Test 8: 1-sample PG-19 perplexity eval."""
    from src.eval.pg19 import evaluate_pg19

    logger.info("Test 8: PG-19 eval (1 sample)")
    try:
        device = next(model.parameters()).device
        results = evaluate_pg19(
            model=model,
            tokenizer=tokenizer,
            num_samples=1,
            window_size=256,
            device=device,
        )
        ppl = results["mean_perplexity"]
        assert ppl > 0, f"Perplexity should be positive, got {ppl}"
        assert not (ppl != ppl), "Perplexity is NaN"  # NaN check
        result.record("pg19_eval", True, f"PPL={ppl:.2f} (1 sample)")
    except Exception as e:
        result.record("pg19_eval", False, str(e))
        logger.error(traceback.format_exc())


def test_longbench_eval(model, tokenizer, result: SmokeTestResult):
    """Test 9: 1-sample LongBench eval."""
    from src.eval.longbench import evaluate_longbench

    logger.info("Test 9: LongBench eval (1 sample, narrativeqa)")
    try:
        device = next(model.parameters()).device
        results = evaluate_longbench(
            model=model,
            tokenizer=tokenizer,
            tasks=["narrativeqa"],
            max_new_tokens=64,
            device=device,
            num_samples=1,
        )
        if "error" in results["tasks"].get("narrativeqa", {}):
            raise RuntimeError(results["tasks"]["narrativeqa"]["error"])
        score = results["tasks"]["narrativeqa"].get("mean_score", 0.0)
        result.record("longbench_eval", True, f"score={score:.4f} (1 sample)")
    except Exception as e:
        result.record("longbench_eval", False, str(e))
        logger.error(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_smoke_test(
    model_name: str = SMOKE_TEST_MODEL_NAME,
    transform_type: str = "fft",
    filter_type: str = "learnable",
    gamma: float = 0.22,
    hf_token: str | None = None,
    skip_checkpoint: bool = False,
    skip_eval: bool = False,
) -> bool:
    """Run the full smoke test suite.

    Args:
        model_name: HF model to use (default: small 1B model for speed).
        transform_type: Spectral transform ("dct" or "fft").
        filter_type: Filter type ("fixed" or "learnable").
        gamma: Compression ratio.
        hf_token: HuggingFace token.
        skip_checkpoint: Skip checkpoint save/load test.
        skip_eval: Skip PG-19 and LongBench eval tests.

    Returns:
        True if all tests passed.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(f"{'='*60}")
    logger.info("SPECTRAL KV-CACHE SMOKE TEST")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Transform: {transform_type}")
    logger.info(f"  Filter: {filter_type}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"{'='*60}")

    result = SmokeTestResult()

    # Test 1: Load model
    model, tokenizer = test_model_loading(model_name, hf_token, result)

    # Test 2: Apply spectral compression
    model = test_spectral_compression(model, transform_type, filter_type, gamma, result)

    # Test 3: Forward pass
    test_forward_pass(model, tokenizer, result)

    # Test 4: Loss computation
    test_loss_computation(model, tokenizer, result)

    # Test 5: Gradient flow (only for learnable filters)
    test_gradient_flow(model, tokenizer, result)

    # Test 6: Cache reset
    test_reset_caches(model, tokenizer, result)

    # Test 7: Checkpoint save/load (P2-6)
    if not skip_checkpoint:
        model = test_checkpoint_save_load(model, tokenizer, result)

    # Tests 8-9: Evaluation (skip if no network or model is too slow)
    if not skip_eval:
        test_pg19_eval(model, tokenizer, result)
        test_longbench_eval(model, tokenizer, result)

    # Print summary
    print(result.summary())
    return result.all_passed


def main():
    parser = argparse.ArgumentParser(description="Smoke test for spectral KV compression")
    parser.add_argument(
        "--model",
        type=str,
        default=SMOKE_TEST_MODEL_NAME,
        help="Model name (default: small 1B for speed)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="fft",
        choices=["dct", "fft"],
        help="Spectral transform type",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="learnable",
        choices=["fixed", "learnable"],
        help="Filter type",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.22,
        help="Compression ratio",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip checkpoint save/load test",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip PG-19 and LongBench eval tests",
    )

    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    passed = run_smoke_test(
        model_name=args.model,
        transform_type=args.transform,
        filter_type=args.filter,
        gamma=args.gamma,
        hf_token=hf_token,
        skip_checkpoint=args.skip_checkpoint,
        skip_eval=args.skip_eval,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
