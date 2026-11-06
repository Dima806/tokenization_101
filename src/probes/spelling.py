"""Spelling probe: how a word tokenizes, and whether its characters are
individually visible to the model.

If a word becomes one or two opaque tokens, the model never receives its
separate letters, so asking it to count or reverse letters asks it to see
inside a chunk it cannot see into.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bpe import BPETokenizer


@dataclass(frozen=True)
class SpellingProbe:
    word: str
    token_ids: list[int]
    pieces: list[str]
    num_characters: int
    num_tokens: int
    characters_visible: bool  # True only if every character is its own token


def probe_spelling(tokenizer: BPETokenizer, word: str) -> SpellingProbe:
    """Tokenize ``word`` and report whether its characters are visible."""
    ids = tokenizer.encode(word)
    pieces = [tokenizer.vocab[i].decode("utf-8", errors="replace") for i in ids]
    visible = len(ids) == len(word) and all(len(piece) == 1 for piece in pieces)
    return SpellingProbe(
        word=word,
        token_ids=ids,
        pieces=pieces,
        num_characters=len(word),
        num_tokens=len(ids),
        characters_visible=visible,
    )


def count_letter(word: str, letter: str) -> int:
    """The true count of ``letter`` in ``word`` (what the model struggles with)."""
    return word.lower().count(letter.lower())
