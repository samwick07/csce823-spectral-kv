"""Spectral transforms and filters for KV-cache compression."""

from .transform import DCTTransform, FFTTransform, SpectralTransform
from .filter import FixedLowPassFilter, LearnableSpectralFilter, SpectralFilter
from .cache import CompressionConfig, SpectralKVCompressor
from .attention import (
    apply_spectral_compression,
    get_spectral_compressors,
    get_spectral_caches,  # backward compat alias for get_spectral_compressors
    get_compression_stats,
    get_learnable_filter_params,
    reset_all_caches,
    CompressedAttention,
)

# Backward compatibility aliases
SpectralKVCache = SpectralKVCompressor

__all__ = [
    "SpectralTransform",
    "DCTTransform",
    "FFTTransform",
    "SpectralFilter",
    "FixedLowPassFilter",
    "LearnableSpectralFilter",
    "CompressionConfig",
    "SpectralKVCompressor",
    "SpectralKVCache",  # backward compat alias
    "apply_spectral_compression",
    "get_spectral_compressors",
    "get_spectral_caches",  # backward compat
    "get_compression_stats",
    "get_learnable_filter_params",
    "reset_all_caches",
    "CompressedAttention",
]
