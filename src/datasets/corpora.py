"""Small, self-contained corpora for training and probing the tokenizer.

Everything here is tiny, permissively authored for this project, and generated
in-memory so nothing needs downloading. ``make data`` caches a frozen copy
under ``data/corpora/``.
"""

from __future__ import annotations

import json
from pathlib import Path

# Several varied English paragraphs (no digits, so the number probe stays crisp),
# rich in the common words and endings BPE loves to merge. The variety lets a
# real vocabulary of a thousand-plus tokens emerge when the text is tiled.
_PARAGRAPHS = [
    "The model does not read the letters in a word. It reads tokens, and a token is a "
    "chunk of characters that a tokenizer learned from a large amount of ordinary text. "
    "The tokenizer looks at the text, counts the pairs of symbols that sit next to each "
    "other, and merges the most common pair into a new symbol. It repeats this simple step "
    "again and again, and the common chunks of the language slowly turn into single tokens. "
    "The word the becomes a token, the ending ing becomes a token, and a space before a "
    "common word becomes a token as well. Because the model only ever sees these tokens, it "
    "cannot see the letters hiding quietly inside them.",
    "People learn to read slowly, one letter at a time, and then almost all at once the "
    "letters seem to disappear and only the meaning stays behind. A fluent reader no longer "
    "notices the separate marks on the page; whole words arrive as single familiar shapes, "
    "and the eye jumps ahead to the end of the sentence before the mouth could ever say it. "
    "We forget how strange this is until we meet a language we cannot read, where every "
    "shape is a small puzzle and nothing arrives as a whole.",
    "On a cold morning in an unfamiliar city, a cup of coffee and a warm piece of bread can "
    "feel like the kindest thing in the world. The streets are quiet, the light is thin and "
    "grey, and the smell of baking drifts out of a small shop on the corner. You do not "
    "share a language with the person behind the counter, so you point and smile, and "
    "somehow that is enough. A good morning needs very few words.",
    "Good work is mostly patience. You try a thing, you watch it fail, you change one small "
    "part, and then you try the whole thing again. Nothing clever usually happens; the "
    "clever part is simply not giving up when the first attempt falls apart in your hands. "
    "Over time the small changes add up, the rough edges wear smooth, and a thing that once "
    "looked impossible begins to feel ordinary and even easy.",
    "Language is older than writing, and writing is younger than we tend to think. For most "
    "of human history people carried their stories in memory and passed them on by mouth, "
    "around a fire, in a field, on a long walk home. The marks we make on paper are only a "
    "recent trick for holding those sounds still, so that a voice can travel across years "
    "and miles and reach a reader who was never there.",
    "When you understand tokens, a whole list of strange machine failures stops looking like "
    "magic. A model that cannot count the letters in a word, that stumbles over long numbers, "
    "or that charges far more for one language than another, is not being lazy or stupid. It "
    "is simply working with the chunks it was handed, and those chunks were shaped by the "
    "text it happened to learn from, long before it ever met your question.",
]
BASE_TEXT = "\n".join(_PARAGRAPHS) + "\n"

# The same meaning across languages and scripts, so the "language tax" is
# measured on matched content rather than anecdotes.
_PARALLEL = {
    "English": "I would like a cup of coffee.",
    "Spanish": "Me gustaría una taza de café.",
    "French": "Je voudrais une tasse de café.",
    "German": "Ich hätte gerne eine Tasse Kaffee.",
    "Portuguese": "Eu gostaria de uma xícara de café.",
    "Ukrainian": "Я хотів би чашку кави.",
    "Greek": "Θα ήθελα ένα φλιτζάνι καφέ.",
    "Hindi": "मुझे एक कप कॉफ़ी चाहिए।",
    "Chinese": "我想要一杯咖啡。",
    "Japanese": "コーヒーを一杯お願いします。",
    "Korean": "커피 한 잔 주세요.",
    "Arabic": "أريد فنجان قهوة.",
}

_SPELLING_WORDS = [
    "strawberry",
    "mississippi",
    "bookkeeper",
    "unbelievable",
    "tokenization",
    "antidisestablishmentarianism",
]

_NUMBER_STRINGS = [
    "1234567",
    "1000000",
    "42424242",
    "3.14159",
    "1234 + 5678",
    "The year 2024 had 366 days.",
]

_CODE_SNIPPETS = [
    "def add(a, b):\n    return a + b",
    "for i in range(10):\n    print(i)",
    "x = {'key': [1, 2, 3], 'flag': True}",
]


def training_corpus(target_chars: int = 200_000) -> str:
    """English training text, tiled to approximately ``target_chars``."""
    if target_chars <= len(BASE_TEXT):
        return BASE_TEXT[:target_chars]
    reps = target_chars // len(BASE_TEXT) + 1
    return (BASE_TEXT * reps)[:target_chars]


def parallel_sentences() -> dict[str, str]:
    """The same sentence across several languages and scripts."""
    return dict(_PARALLEL)


def spelling_words() -> list[str]:
    """Words chosen to expose character-counting failures."""
    return list(_SPELLING_WORDS)


def number_strings() -> list[str]:
    """Numbers and arithmetic chosen to expose place-value splits."""
    return list(_NUMBER_STRINGS)


def code_snippets() -> list[str]:
    """Short code snippets, a token-hungry domain."""
    return list(_CODE_SNIPPETS)


def cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "corpora"


def main() -> None:
    """Write a frozen copy of every corpus to ``data/corpora/``."""
    out = cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_corpus.txt").write_text(training_corpus(), encoding="utf-8")
    (out / "parallel_sentences.json").write_text(
        json.dumps(parallel_sentences(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "probes.json").write_text(
        json.dumps(
            {
                "spelling_words": spelling_words(),
                "number_strings": number_strings(),
                "code_snippets": code_snippets(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Cached corpora in {out}")


if __name__ == "__main__":
    main()
