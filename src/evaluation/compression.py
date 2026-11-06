"""The compression view: tokens per character.

BPE is a compression algorithm, so tokens-per-character is a natural measure of
how hard a piece of text is to tokenize. Fewer tokens per character means
better compression, which is exactly what English enjoys and other languages
do not. (Connects to information_theory_101.)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bpe import BPETokenizer


def tokens_per_character(tokenizer: BPETokenizer, text: str) -> float:
    """Tokens divided by characters (lower = better compression)."""
    if not text:
        return 0.0
    return len(tokenizer.encode(text)) / len(text)


def characters_per_token(tokenizer: BPETokenizer, text: str) -> float:
    """Characters divided by tokens (higher = better compression)."""
    ids = tokenizer.encode(text)
    if not ids:
        return 0.0
    return len(text) / len(ids)


@dataclass(frozen=True)
class CompressionRow:
    label: str
    n_chars: int
    n_tokens: int
    tokens_per_char: float


def compression_summary(tokenizer: BPETokenizer, texts: dict[str, str]) -> list[CompressionRow]:
    """Per-text compression rows, ready to tabulate or plot."""
    rows: list[CompressionRow] = []
    for label, text in texts.items():
        n_tokens = len(tokenizer.encode(text))
        ratio = n_tokens / len(text) if text else 0.0
        rows.append(
            CompressionRow(
                label=label, n_chars=len(text), n_tokens=n_tokens, tokens_per_char=ratio
            )
        )
    return rows
