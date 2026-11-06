"""Digit tokenization splits across place values (demonstrated)."""

from __future__ import annotations

from src.bpe import BPETokenizer
from src.probes.numbers import place_value_groups, probe_numbers


def test_place_value_grouping_helper() -> None:
    assert place_value_groups("1234567") == ["1", "234", "567"]
    assert place_value_groups("12") == ["12"]
    assert place_value_groups("1000000") == ["1", "000", "000"]
    assert place_value_groups("") == []


def test_digits_do_not_respect_place_value(tokenizer: BPETokenizer) -> None:
    numbers = ["1234567", "1000000", "42424242", "1234 + 5678"]
    probes = [probe_numbers(tokenizer, n) for n in numbers]
    # The whole point: the tokenizer cuts digits across human place-value groups.
    assert all(not p.respects_place_value for p in probes)
    # And a multi-digit number is genuinely split into several pieces.
    assert any(len(p.digit_pieces) > 1 for p in probes)
