"""Non-English costs materially more tokens for matched content (asserted)."""

from __future__ import annotations

from src.bpe import BPETokenizer
from src.datasets.corpora import parallel_sentences
from src.probes.multilingual import language_tax


def test_non_english_costs_more_tokens(tokenizer: BPETokenizer) -> None:
    taxes = {t.language: t for t in language_tax(tokenizer, parallel_sentences())}
    english = taxes["English"].num_tokens
    assert english > 0
    # Non-Latin scripts fragment into many more tokens for the same meaning.
    for language in ["Ukrainian", "Chinese", "Japanese", "Hindi", "Arabic"]:
        assert taxes[language].num_tokens > english, language
        assert taxes[language].ratio_to_english > 1.0, language


def test_english_ratio_is_one(tokenizer: BPETokenizer) -> None:
    taxes = {t.language: t for t in language_tax(tokenizer, parallel_sentences())}
    assert taxes["English"].ratio_to_english == 1.0
