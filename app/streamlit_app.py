"""Tokenization 101 — interactive playground.

Four tools, all powered by the from-scratch BPE tokenizer:
  1. Live tokenizer      — see any text split into tokens.
  2. Spelling probe      — see whether a word's letters are visible.
  3. Number probe        — see digits split across place values.
  4. Language tax meter  — see token cost across languages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.bpe import BPETokenizer
from src.datasets.corpora import parallel_sentences, training_corpus
from src.probes.multilingual import language_tax
from src.probes.numbers import probe_numbers
from src.probes.spelling import count_letter, probe_spelling


@st.cache_resource
def load_tokenizer() -> BPETokenizer:
    return BPETokenizer().train(training_corpus(80_000), vocab_size=512)


def render_tokens(pieces: list[str]) -> str:
    chips = [
        f"<span style='background:#eef;border:1px solid #99a;border-radius:4px;"
        f"padding:1px 4px;margin:1px;display:inline-block'>"
        f"{piece.replace(' ', '␣').replace(chr(10), '⏎')}</span>"
        for piece in pieces
    ]
    return "".join(chips)


st.set_page_config(page_title="Tokenization 101", page_icon="🍓", layout="wide")
st.title("🍓 Tokenization 101")
st.caption("The model sees tokens, not characters. Build the intuition here.")

tokenizer = load_tokenizer()

live, spelling, numbers, tax = st.tabs(
    ["Live tokenizer", "Spelling probe", "Number probe", "Language tax"]
)

with live:
    text = st.text_area("Text", "The strawberry costs $12.50 in café.", key="live")
    ids = tokenizer.encode(text)
    pieces = tokenizer.pieces(text)
    col_a, col_b = st.columns(2)
    col_a.metric("Characters", len(text))
    col_b.metric("Tokens", len(ids))
    st.markdown(render_tokens(pieces), unsafe_allow_html=True)
    st.caption(f"round-trip ok: {tokenizer.decode(ids) == text}")

with spelling:
    word = st.text_input("Word", "strawberry", key="spell")
    letter = st.text_input("Letter to count", "r", key="letter")
    probe = probe_spelling(tokenizer, word)
    st.markdown(render_tokens(probe.pieces), unsafe_allow_html=True)
    st.write(
        f"**{probe.num_characters}** characters became **{probe.num_tokens}** tokens. "
        f"Characters individually visible to the model: **{probe.characters_visible}**."
    )
    if letter:
        st.write(
            f"True count of '{letter}' in '{word}': **{count_letter(word, letter)}** "
            "— the model is reasoning over opaque chunks, not these letters."
        )

with numbers:
    number_text = st.text_input("Number or expression", "1234 + 5678", key="num")
    nprobe = probe_numbers(tokenizer, number_text)
    st.markdown(render_tokens(nprobe.pieces), unsafe_allow_html=True)
    st.write(f"Digit tokens: `{nprobe.digit_pieces}`")
    st.write(f"Place-value grouping: `{nprobe.place_value}`")
    st.write(f"Respects place value: **{nprobe.respects_place_value}**")

with tax:
    sentence_sets = parallel_sentences()
    rows = language_tax(tokenizer, sentence_sets)
    frame = pd.DataFrame(
        [
            {
                "language": row.language,
                "text": row.text,
                "tokens": row.num_tokens,
                "ratio_to_English": round(row.ratio_to_english, 2),
            }
            for row in rows
        ]
    ).sort_values("tokens")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.bar_chart(frame.set_index("language")["tokens"])
