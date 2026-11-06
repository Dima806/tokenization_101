"""The money view: per-token cost across languages.

API pricing is per token, so the language tax is also a cost tax: the same
meaning in another language costs more because it is more tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bpe import BPETokenizer


def token_cost(n_tokens: int, usd_per_1k: float = 0.01) -> float:
    """Cost of ``n_tokens`` at ``usd_per_1k`` dollars per 1,000 tokens."""
    return n_tokens / 1000.0 * usd_per_1k


@dataclass(frozen=True)
class CostRow:
    language: str
    n_tokens: int
    usd: float
    ratio_to_base: float


def cost_across_languages(
    tokenizer: BPETokenizer,
    sentences: dict[str, str],
    usd_per_1k: float = 0.01,
    base: str = "English",
) -> list[CostRow]:
    """Per-language token count, dollar cost, and ratio to the base language."""
    base_tokens = len(tokenizer.encode(sentences[base]))
    rows: list[CostRow] = []
    for language, text in sentences.items():
        n_tokens = len(tokenizer.encode(text))
        ratio = n_tokens / base_tokens if base_tokens else 0.0
        rows.append(
            CostRow(
                language=language,
                n_tokens=n_tokens,
                usd=token_cost(n_tokens, usd_per_1k),
                ratio_to_base=ratio,
            )
        )
    return rows
