"""Fragmentation and coverage: how many tokens a tokenizer spends per word.

A tokenizer that fragments text into many short tokens has poor coverage of
that domain or language. Tokens-per-word makes the fragmentation concrete.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from src.bpe import BPETokenizer

# A "word" is a run of letters, a run of digits, or any single non-space symbol.
_WORD = re.compile(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]")


def word_count(text: str) -> int:
    """Count words (letters/digits runs and standalone symbols)."""
    return len(_WORD.findall(text))


def fragmentation(tokenizer: BPETokenizer, text: str) -> float:
    """Tokens per word (1.0 is ideal; higher means more fragmentation)."""
    words = word_count(text)
    if words == 0:
        return 0.0
    return len(tokenizer.encode(text)) / words


@dataclass(frozen=True)
class CoverageRow:
    label: str
    n_words: int
    n_tokens: int
    tokens_per_word: float


def coverage_report(tokenizer: BPETokenizer, texts: dict[str, str]) -> list[CoverageRow]:
    """Per-text fragmentation rows across domains or languages."""
    rows: list[CoverageRow] = []
    for label, text in texts.items():
        words = word_count(text)
        n_tokens = len(tokenizer.encode(text))
        ratio = n_tokens / words if words else 0.0
        rows.append(
            CoverageRow(label=label, n_words=words, n_tokens=n_tokens, tokens_per_word=ratio)
        )
    return rows
