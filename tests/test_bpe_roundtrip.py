"""encode(decode(x)) == x: byte-level BPE round-trips any input exactly."""

from __future__ import annotations

import pytest

from src.bpe import BPETokenizer, build_base_vocab, decode, encode

ROUNDTRIP_INPUTS = [
    "",
    "a",
    "hello world",
    "strawberry",
    "  leading and  double  spaces ",
    "Numbers 1234567890 and symbols !@#$%^&*()",
    "CamelCase and snake_case and kebab-case",
    "Köln, café, naïve, façade",  # latin diacritics
    "Привіт, світе",  # cyrillic (ukrainian)
    "こんにちは世界",  # japanese
    "你好，世界",  # chinese
    "안녕하세요",  # korean
    "مرحبا بالعالم",  # arabic
    "emoji 🍓🎉👩‍💻 family",  # emoji incl. zero-width joiner
    "mixed English and 日本語 and русский together",
]


@pytest.mark.parametrize("text", ROUNDTRIP_INPUTS)
def test_decode_encode_roundtrip(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_untrained_tokenizer_still_roundtrips() -> None:
    # With no merges, encoding is just the UTF-8 bytes, and still reversible.
    tok = BPETokenizer()
    for text in ROUNDTRIP_INPUTS:
        assert tok.decode(tok.encode(text)) == text


def test_module_level_encode_decode() -> None:
    tok = BPETokenizer().train("the theory of the theme of the theatre", vocab_size=300)
    ids = encode("the theme", tok.merges)
    assert decode(ids, tok.vocab) == "the theme"


def test_base_vocab_covers_every_byte() -> None:
    base = build_base_vocab()
    assert len(base) == 256
    assert base[65] == b"A"
