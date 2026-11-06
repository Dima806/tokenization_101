"""Shared fixtures: one trained tokenizer for the whole test session."""

from __future__ import annotations

import pytest

from src.bpe import BPETokenizer
from src.datasets.corpora import training_corpus


@pytest.fixture(scope="session")
def tokenizer() -> BPETokenizer:
    """A small BPE tokenizer trained once on the English corpus."""
    return BPETokenizer().train(training_corpus(target_chars=40_000), vocab_size=512)
