"""Multilingual probe: token counts for the same meaning across languages.

Tokenizers are trained on English-heavy text, so English compresses into few
tokens while other languages (especially non-Latin scripts) fragment into many
more for the identical meaning — the "language tax".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bpe import BPETokenizer


@dataclass(frozen=True)
class LanguageTax:
    language: str
    text: str
    num_tokens: int
    ratio_to_english: float


def language_tax(
    tokenizer: BPETokenizer, sentences: dict[str, str], base: str = "English"
) -> list[LanguageTax]:
    """Token count per language, with the ratio to the ``base`` language."""
    base_tokens = len(tokenizer.encode(sentences[base]))
    rows: list[LanguageTax] = []
    for language, text in sentences.items():
        count = len(tokenizer.encode(text))
        ratio = count / base_tokens if base_tokens else 0.0
        rows.append(
            LanguageTax(language=language, text=text, num_tokens=count, ratio_to_english=ratio)
        )
    return rows
