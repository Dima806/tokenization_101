"""Number probe: how digit strings split into tokens, and whether the split
respects place value.

Humans read numbers in groups of three from the right (place value). A
tokenizer trained on text splits digits however they happened to co-occur,
cutting across those groups, so the model never receives numbers in a form that
respects how addition works.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bpe import BPETokenizer


def place_value_groups(digits: str) -> list[str]:
    """Group ``digits`` into threes from the right, e.g. ``"1234567"`` ->
    ``["1", "234", "567"]``."""
    reversed_digits = digits[::-1]
    chunks = [reversed_digits[i : i + 3][::-1] for i in range(0, len(reversed_digits), 3)]
    return chunks[::-1]


@dataclass(frozen=True)
class NumberProbe:
    text: str
    token_ids: list[int]
    pieces: list[str]
    digit_pieces: list[str]
    place_value: list[str]
    respects_place_value: bool


def probe_numbers(tokenizer: BPETokenizer, text: str) -> NumberProbe:
    """Tokenize ``text`` and compare its digit split to place-value grouping."""
    ids = tokenizer.encode(text)
    pieces = [tokenizer.vocab[i].decode("utf-8", errors="replace") for i in ids]
    digit_pieces = [piece for piece in pieces if piece.isdigit()]
    digits = "".join(ch for ch in text if ch.isdigit())
    groups = place_value_groups(digits)
    respects = bool(digits) and digit_pieces == groups
    return NumberProbe(
        text=text,
        token_ids=ids,
        pieces=pieces,
        digit_pieces=digit_pieces,
        place_value=groups,
        respects_place_value=respects,
    )
