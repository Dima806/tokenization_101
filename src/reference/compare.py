"""Compare the from-scratch tokenizer against production references.

``tiktoken`` (GPT family) and HF ``tokenizers`` are used ONLY to validate and
contextualise the from-scratch BPE — never as the implementation. Loading a
reference encoding may require a one-time download, so these helpers import and
fetch lazily (inside the functions) and are not exercised by the offline tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.bpe import BPETokenizer


def get_reference_encoding(name: str = "gpt2") -> Any:  # noqa: ANN401 - tiktoken is untyped
    """Return a tiktoken encoding (downloads its vocab on first use)."""
    import tiktoken

    return tiktoken.get_encoding(name)


def try_get_reference_encoding(name: str = "gpt2") -> Any:  # noqa: ANN401 - tiktoken is untyped
    """Return a tiktoken encoding, or ``None`` if it cannot be loaded.

    Useful in notebooks so a missing network connection degrades to the
    from-scratch tokenizer instead of raising.
    """
    try:
        return get_reference_encoding(name)
    except Exception:
        return None


def reference_token_count(text: str, name: str = "gpt2") -> int:
    """Token count for ``text`` under a named reference encoding."""
    encoding = get_reference_encoding(name)
    return len(encoding.encode(text))


@dataclass(frozen=True)
class Comparison:
    text: str
    ours: int
    reference: int
    reference_name: str


def compare_counts(tokenizer: BPETokenizer, text: str, name: str = "gpt2") -> Comparison:
    """Compare our token count to a reference tokenizer's for ``text``."""
    return Comparison(
        text=text,
        ours=len(tokenizer.encode(text)),
        reference=reference_token_count(text, name),
        reference_name=name,
    )
