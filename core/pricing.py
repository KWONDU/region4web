"""LLM pricing per token.

Prices in USD per 1M tokens. Local models (vLLM, HuggingFace names) have no
entry in PRICING; ``calculate_cost`` returns None for those so the caller can
omit the cost field from its trajectory dumps entirely.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# USD per 1M tokens. Keys:
#   input         — fresh (uncached) input
#   cached_input  — prompt-cache hit (OpenAI: 50% off, DeepSeek/Gemini: ~75% off)
#   output        — completion; for OpenAI reasoning models this INCLUDES
#                   reasoning_tokens, which are billed at the output rate
PRICING: Dict[str, Dict[str, float]] = {
    # -- OpenAI --
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    # -- DeepSeek --
    "deepseek-chat": {"input": 0.28, "cached_input": 0.028, "output": 0.42},
    "deepseek-reasoner": {"input": 0.28, "cached_input": 0.028, "output": 0.42},
    # -- Google / Gemini --
    "gemini-3.1-flash": {"input": 0.75, "cached_input": 0.75, "output": 4.50},
    "gemini-3.1-flash-live-preview": {
        "input": 0.75,
        "cached_input": 0.75,
        "output": 4.50,
    },
    "gemini-3.1-flash-lite-preview": {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 1.50,
    },
}

# Batch API discount (OpenAI: 50% off)
BATCH_DISCOUNT: float = 0.5


def calculate_cost(
    usage: Dict[str, Any],
    model: str,
    batch: bool = False,
) -> Optional[Dict[str, float]]:
    """Calculate cost from a usage dict.

    Args:
        usage: ``{"prompt_tokens": int, "completion_tokens": int,
                  "cached_tokens": int (OpenAI) or
                  "prompt_cache_hit_tokens": int (DeepSeek), ...}``
        model: model name without provider prefix (e.g. ``"gpt-5.1"``).
        batch: if True, apply the batch-API discount.

    Returns:
        ``{"input_cost", "cached_input_cost", "output_cost", "total_cost"}``
        in USD, or **``None`` when the model is not in PRICING**. None is the
        "unknown unit price" signal — callers that persist cost to disk
        should leave the cost field out entirely in that case (do NOT write
        0.0; that would conflate "free local compute" with "unknown price").
    """
    prices = PRICING.get(model)
    if prices is None:
        return None

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    cached_tokens = (
        usage.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0
    )
    cached_tokens = min(cached_tokens, prompt_tokens)
    fresh_tokens = max(prompt_tokens - cached_tokens, 0)
    output_tokens = (
        usage.get("completion_tokens", 0) or 0
    )  # includes reasoning_tokens for OpenAI

    cached_rate = prices.get("cached_input", prices["input"])
    input_cost = fresh_tokens * prices["input"] / 1_000_000
    cache_cost = cached_tokens * cached_rate / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000

    if batch:
        input_cost *= BATCH_DISCOUNT
        cache_cost *= BATCH_DISCOUNT
        output_cost *= BATCH_DISCOUNT

    return {
        "input_cost": input_cost,
        "cached_input_cost": cache_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + cache_cost + output_cost,
    }
