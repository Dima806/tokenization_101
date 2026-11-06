"""The from-scratch BPE reproduces a known, hand-traced merge sequence, and
(when available) agrees with a production tokenizer that it round-trips.
"""

from __future__ import annotations

import pytest

from src.bpe import BPETokenizer, train_bpe

# Hand-traced known answer for the classic BPE example corpus "aaabdaaabac"
# (bytes a=97, b=98, c=99, d=100), ties broken by the lexicographically
# smallest pair so the result is fully determined:
#   step 1: (97, 97)   -> 256  b"aa"    (count 4)
#   step 2: (97, 98)   -> 257  b"ab"    (count 2; beats (256, 97) on the tie)
#   step 3: (256, 257) -> 258  b"aaab"  (count 2)
#   step 4: best pair now occurs only once -> training stops
EXPECTED_MERGES = [((97, 97), 256), ((97, 98), 257), ((256, 257), 258)]
EXPECTED_PIECES = {256: b"aa", 257: b"ab", 258: b"aaab"}


def test_from_scratch_merges_match_known_answer() -> None:
    merges, vocab = train_bpe("aaabdaaabac", vocab_size=300)
    assert list(merges.items()) == EXPECTED_MERGES
    for new_id, piece in EXPECTED_PIECES.items():
        assert vocab[new_id] == piece


def test_training_is_deterministic() -> None:
    corpus = "the quick brown fox the quick brown fox the lazy dog"
    first, _ = train_bpe(corpus, vocab_size=320)
    second, _ = train_bpe(corpus, vocab_size=320)
    assert first == second


def test_agrees_with_tiktoken_when_available() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    try:
        encoding = tiktoken.get_encoding("gpt2")
    except Exception as exc:  # offline / no cached vocab
        pytest.skip(f"reference tokenizer unavailable: {exc}")

    text = "The quick brown fox jumps over the lazy dog."
    # Both are byte-level and must reconstruct the input exactly.
    assert encoding.decode(encoding.encode(text)) == text
    tok = BPETokenizer().train(text * 20, vocab_size=400)
    assert tok.decode(tok.encode(text)) == text
