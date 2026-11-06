"""Train byte-pair encoding (BPE) from scratch.

The whole algorithm in one place: represent the text as raw UTF-8 bytes, count
every adjacent pair of symbols, merge the most frequent pair into a brand-new
symbol, record the merge rule, and repeat until the vocabulary reaches the
target size. The vocabulary is *learned from frequency*, not designed.

Working at the byte level (0..255 as the starting alphabet) means any Unicode
input is representable and round-trips exactly, which is how production
byte-level BPE (e.g. GPT-2) works.
"""

from __future__ import annotations

from collections import Counter

# A pair of adjacent token ids, the id it merges into, and the two lookup tables.
Pair = tuple[int, int]
Merges = dict[Pair, int]
Vocab = dict[int, bytes]


def get_stats(ids: list[int]) -> Counter[Pair]:
    """Count how often each adjacent pair of ids occurs in ``ids``."""
    counts: Counter[Pair] = Counter()
    for pair in zip(ids[:-1], ids[1:], strict=False):
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def best_pair(stats: Counter[Pair]) -> Pair:
    """Pick the pair to merge: most frequent, ties broken by the
    lexicographically smallest pair so training is fully deterministic.
    """
    return max(stats, key=lambda p: (stats[p], -p[0], -p[1]))


def build_base_vocab() -> Vocab:
    """The starting vocabulary: one token per possible byte value."""
    return {i: bytes([i]) for i in range(256)}


def train_bpe(text: str, vocab_size: int, *, verbose: bool = False) -> tuple[Merges, Vocab]:
    """Train BPE on ``text`` until the vocabulary reaches ``vocab_size``.

    Returns the learned ``merges`` (in merge order) and the ``vocab`` mapping
    token ids to their byte sequences. Stops early if no pair repeats, since
    merging a one-off pair buys no compression.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256 (one id per byte value)")

    ids = list(text.encode("utf-8"))
    merges: Merges = {}
    vocab: Vocab = build_base_vocab()

    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats:
            break
        pair = best_pair(stats)
        if stats[pair] < 2:
            break  # nothing repeats anymore; stop
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {i + 1}: {pair} -> {new_id} ({vocab[new_id]!r})")

    return merges, vocab
