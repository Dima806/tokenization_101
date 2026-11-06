"""Inspect a learned BPE vocabulary and the order of its merges."""

from __future__ import annotations

from dataclasses import dataclass

from src.bpe.train import Merges, Vocab, build_base_vocab


def build_vocab(merges: Merges) -> Vocab:
    """Reconstruct the id -> bytes vocabulary from the merge rules alone."""
    vocab = build_base_vocab()
    for (first, second), new_id in merges.items():
        vocab[new_id] = vocab[first] + vocab[second]
    return vocab


def render_token(token: bytes) -> str:
    """Render a token's bytes for display, making spaces visible as ``·``."""
    return token.decode("utf-8", errors="replace").replace(" ", "·")


@dataclass(frozen=True)
class MergeRow:
    """One row of the merge table: when and what was merged."""

    rank: int
    new_id: int
    pair: tuple[int, int]
    piece: str


def merge_table(merges: Merges, vocab: Vocab | None = None) -> list[MergeRow]:
    """Return the merges as ordered rows, for printing or plotting."""
    resolved = vocab if vocab is not None else build_vocab(merges)
    return [
        MergeRow(rank=rank, new_id=new_id, pair=pair, piece=render_token(resolved[new_id]))
        for rank, (pair, new_id) in enumerate(merges.items())
    ]
