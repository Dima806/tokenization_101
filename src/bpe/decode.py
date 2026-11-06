"""Decode BPE token ids back into text (the inverse of :func:`encode`)."""

from __future__ import annotations

from src.bpe.train import Vocab


def decode(ids: list[int], vocab: Vocab) -> str:
    """Turn token ids back into a string by concatenating their bytes.

    Because every token id maps to a concrete byte sequence, decoding is exact:
    ``decode(encode(text)) == text`` for any string. ``errors="replace"`` only
    matters if a caller hands in an arbitrary mid-sequence slice of ids.
    """
    data = b"".join(vocab[i] for i in ids)
    return data.decode("utf-8", errors="replace")
