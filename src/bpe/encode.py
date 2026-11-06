"""Encode text into BPE token ids by applying the learned merge rules."""

from __future__ import annotations

from src.bpe.train import Merges, get_stats, merge


def encode(text: str, merges: Merges) -> list[int]:
    """Tokenize ``text`` by repeatedly applying the earliest-learned merge.

    Start from the raw UTF-8 bytes and, at each step, apply the merge that was
    learned first (lowest new id) among the pairs currently present. This
    reproduces the order in which the merges were discovered during training.
    """
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # The pair whose merge rule has the lowest rank (learned earliest).
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # no remaining pair can be merged
        ids = merge(ids, pair, merges[pair])
    return ids
