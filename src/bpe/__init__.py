"""From-scratch byte-level BPE: training, encoding, decoding, inspection.

The :class:`BPETokenizer` ties the pieces together for convenient use in
notebooks, probes and the app; the underlying functions stay importable
directly (``from src.bpe import train_bpe, encode, decode``).
"""

from __future__ import annotations

from src.bpe.decode import decode
from src.bpe.encode import encode
from src.bpe.train import (
    Merges,
    Pair,
    Vocab,
    best_pair,
    build_base_vocab,
    get_stats,
    merge,
    train_bpe,
)
from src.bpe.vocab import MergeRow, build_vocab, merge_table, render_token


class BPETokenizer:
    """A byte-level BPE tokenizer built entirely from scratch."""

    def __init__(self, merges: Merges | None = None, vocab: Vocab | None = None) -> None:
        self.merges: Merges = merges if merges is not None else {}
        self.vocab: Vocab = vocab if vocab is not None else build_vocab(self.merges)

    def train(self, text: str, vocab_size: int, *, verbose: bool = False) -> BPETokenizer:
        """Learn merges from ``text`` and keep them on this tokenizer."""
        self.merges, self.vocab = train_bpe(text, vocab_size, verbose=verbose)
        return self

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids."""
        return encode(text, self.merges)

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text."""
        return decode(ids, self.vocab)

    def pieces(self, text: str) -> list[str]:
        """The surface string of each token, for display."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]

    def merge_table(self) -> list[MergeRow]:
        """The learned merges as ordered, printable rows."""
        return merge_table(self.merges, self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


__all__ = [
    "BPETokenizer",
    "MergeRow",
    "Merges",
    "Pair",
    "Vocab",
    "best_pair",
    "build_base_vocab",
    "build_vocab",
    "decode",
    "encode",
    "get_stats",
    "merge",
    "merge_table",
    "render_token",
    "train_bpe",
]
