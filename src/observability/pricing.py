"""Token pricing (USD per 1M tokens) and per-query cost calculation.

The built-in table is a convenience default; verify against the provider's current price list
or override with LLM_INPUT_PRICE_PER_1M / LLM_OUTPUT_PRICE_PER_1M. Unknown models cost 0 for
tokens (local Ollama models are free per token; use LOCAL_COMPUTE_COST_PER_HOUR for infra).
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_TABLE_PER_1M: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("anthropic", "claude-3-5-sonnet-20241022"): (3.00, 15.00),
    ("google", "gemini-1.5-pro"): (1.25, 5.00),
}


@dataclass(frozen=True)
class CostBreakdown:
    input_cost_usd: float
    output_cost_usd: float
    infra_cost_usd: float
    input_price_per_1m: float
    output_price_per_1m: float

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd + self.infra_cost_usd


def resolve_prices(
    provider: str,
    model: str,
    input_override: float | None = None,
    output_override: float | None = None,
) -> tuple[float, float]:
    default_in, default_out = PRICE_TABLE_PER_1M.get((provider.lower(), model), (0.0, 0.0))
    return (
        default_in if input_override is None else input_override,
        default_out if output_override is None else output_override,
    )


def compute_cost(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    local_compute_seconds: float = 0.0,
    input_price_per_1m: float | None = None,
    output_price_per_1m: float | None = None,
    compute_cost_per_hour: float = 0.0,
) -> CostBreakdown:
    price_in, price_out = resolve_prices(provider, model, input_price_per_1m, output_price_per_1m)
    return CostBreakdown(
        input_cost_usd=prompt_tokens / 1_000_000 * price_in,
        output_cost_usd=completion_tokens / 1_000_000 * price_out,
        infra_cost_usd=local_compute_seconds / 3600 * compute_cost_per_hour,
        input_price_per_1m=price_in,
        output_price_per_1m=price_out,
    )
