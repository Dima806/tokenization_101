# Project: tokenization_101

> A hands-on, build-from-scratch introduction to **tokenization** — the invisible
> first step every language model depends on and almost nobody understands.

## Identity

Byte-pair encoding (BPE) built from scratch, on CPU, in ~50 lines. Ships as a
GitHub repo with **6 notebooks**, a **Streamlit app**, and a **`src/` library**,
all runnable in a 2-CPU / 8 GB GitHub Codespace with **no GPU**.

Narrative build-up (the notebook order is the argument):
`text is not characters` → `BPE from scratch` → `why models can't spell` →
`numbers and arithmetic` → `the language tax` → `decision framework`.

## Thesis (say this in one breath)

A model does not read letters or words — it reads **tokens**: sub-word chunks a
compression algorithm (BPE) carved out of a training corpus. **Whatever the
tokenizer hides, the model cannot see.** That single preprocessing choice explains
a list of "mysterious" failures:

- **Can't spell / count letters.** `"strawberry"` is 1–2 tokens (production tokenizer),
  not 10 letters, so counting its r's asks the model to see inside an opaque chunk.
- **Wobbly arithmetic.** Digits split into tokens that cut across place values, so
  numbers never arrive in a place-value-respecting form → route to a tool.
- **The language tax.** Tokenizers are trained on English-heavy corpora, so other
  languages (especially non-Latin scripts) fragment into **2–3× more tokens** for
  the same meaning → 2–3× the cost (pricing is per token), context fills faster,
  effective sequences are shorter.

BPE is compression (ties to `information_theory_101`). The from-scratch build plus
three probes turn these from anecdotes into **measured** claims.

## Audience

Data scientists / engineers who have only ever *called* a tokenizer and want to
understand what it does and why its choices matter — and anyone puzzled by spelling,
counting, arithmetic, or multilingual-cost failures. This is the step *before*
`transformers_101` and `embeddings_101`.

## Stack & constraints

- **Python 3.11+**, package manager **`uv`** (no pip), **CPU only**, 2-CPU / 8 GB.
- Runtime deps: `tiktoken`, `tokenizers`, `regex`, `numpy`, `pandas`, `matplotlib`,
  `plotly`, `streamlit`, `pydantic`, `pydantic-settings`, `pyyaml`.
- Dev: `pytest` (+`pytest-cov`), `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
- **`tiktoken` and `tokenizers` are for validation/comparison ONLY.** The BPE
  tokenizer is built from scratch — never import them as the implementation.
- Everything open-source. Each notebook runs in **< 3 minutes** on CPU.

## Repository structure

```
config/settings.yaml          # vocab sizes, corpora, languages, seeds (pydantic-settings)
src/
  config.py                   # typed settings loader (pydantic)
  artifacts.py                # save_figure → outputs/figures/*.png, save_json → outputs/data/*.json
  bpe/        train.py        # bytes → pair counts → merges → vocab (the ~50 lines)
              encode.py       # apply learned merges to new text
              decode.py       # tokens → text (round-trip inverse of encode)
              vocab.py        # inspect learned vocab + merge order
              __init__.py     # BPETokenizer (ties train/encode/decode/inspect together)
  reference/  compare.py      # validate vs tiktoken; try_get_reference_encoding() is offline-safe
  probes/     spelling.py     # how a word tokenizes; are its characters visible?
              numbers.py      # how digit strings split; place-value alignment
              multilingual.py # token counts per language for matched content
  datasets/   corpora.py      # enriched English corpus + parallel sentences (12 languages; Ukrainian for Cyrillic)
  evaluation/ compression.py  # tokens-per-character (the compression view)
              vocab_coverage.py # OOV / fragmentation across domains & languages
              cost.py         # per-language token-cost model
  visualisation.py            # shared matplotlib helpers (headless-safe; inline in notebooks)
app/streamlit_app.py          # live tokenizer, spelling/number probes, language-tax meter
notebooks/01..06_*.ipynb      # the narrative; executed in place, save PNG + JSON artifacts
tests/                        # roundtrip, vs-reference, numbers, language-tax (25 tests)
data/corpora/                 # frozen corpus cache (make data)
outputs/figures/*.png         # every notebook figure (9)
outputs/data/*.json           # every textual/numerical input & result (11)
```

## Key concepts (the mental model to preserve everywhere)

- **The model sees tokens, not characters.** This is the load-bearing idea; every
  explanation reduces to it.
- **BPE algorithm, in full:** represent the corpus as bytes → count every adjacent
  pair → merge the most frequent pair into a new symbol → record the merge rule →
  repeat until the vocab hits the target size. The vocabulary is *learned from
  frequency, not designed*; watch chunks like `"ing"`, `"the"`, `" strawberry"`
  crystallize as merges accumulate.
- **Byte-level** implementation so arbitrary Unicode input round-trips (matches how
  production byte-level BPE works).
- **Vocab size is a trade-off:** larger vocab → shorter sequences but more embedding
  parameters and worse rare-token handling.
- Exact token counts depend on *which* tokenizer; always report against a **named**
  reference. The qualitative claims (chunks hide characters, digits split,
  non-English costs more) hold across all of them.

## Notebooks (the argument, in order)

1. **01_text_is_not_characters** — the reframe: same string as chars / bytes / tokens;
   establish that the model receives the token sequence. Introduce tokens-per-char.
2. **02_bpe_from_scratch** *(showpiece)* — build BPE end to end; encode/decode with a
   tested round-trip; inspect vocab & merge order; validate against tiktoken.
3. **03_why_models_cant_spell** *(explanatory core)* — tokenize `"strawberry"` & friends;
   show characters aren't individually present; leading-space / capitalization brittleness.
4. **04_numbers_and_arithmetic** — digits split across place values; contrast digit-level;
   why tool use fixes arithmetic.
5. **05_the_language_tax** *(equity/economics crux)* — same meaning across languages/scripts;
   translate token counts into money, context, and modeling quality; report the tax as a ratio.
6. **06_decision_framework** — the practical guide; ships the from-scratch trainer/encoder/
   decoder + the three probes so the reader leaves with working code.

## Commands (see `Makefile`)

```
make setup      # first-time: install uv, uv sync --all-extras, register Jupyter kernel
make sync       # sync deps (incl. dev extras)
make data       # build/cache training corpus + parallel sentences → data/
make lint       # ruff format + ruff check --fix + ty check src
make test       # pytest           (make test-cov for coverage)
make notebooks  # execute all notebooks end-to-end (< 3 min each)
make run        # Streamlit app     make lab = JupyterLab
make ci         # sync + lint + test        make dev = lint + test (fast loop)
```

Project is a **non-packaged uv project** (`[tool.uv] package = false`): code is
imported as `src.*` from the repo root (pytest uses `pythonpath = ["."]`). Run
module entrypoints as `python -m src.datasets.corpora`. Python is pinned to
**3.12** via `.python-version`. `make notebooks` executes each notebook **in
place** and writes its figures (PNG) and data (JSON) to `outputs/`.

## Data

Enriched mixed-English training corpus (several varied paragraphs, **no digits** so
the number probe stays crisp) — enough for a real vocab of 1,000+ tokens to emerge,
still instant on CPU — plus **parallel multilingual sentences** (the same sentence in
**12 languages/scripts**; Ukrainian covers Cyrillic, plus Greek, Hindi, Chinese,
Japanese, Korean, Arabic, …, so the tax is measured on matched meaning) and curated
probe inputs (spelling words, number strings, code). Everything is small, cached,
and permissively sourced, with a **frozen copy committed** (`make data`) so nothing
downloads.

## Success criteria (these are asserted/tested, not decorative)

- **Round-trip:** `encode(decode(x)) == x` for arbitrary input — asserted.
- **Vs reference:** from-scratch merges/vocab reproduce the expected result on a
  fixed corpus — asserted.
- **Numbers probe:** digit tokenization splits across place values — demonstrated.
- **Language tax:** non-English costs materially more tokens for matched content —
  asserted as a ratio.
- `make lint` and `make test` pass with **zero errors**; notebooks run **< 3 min** each.

## House rules (do / don't)

- **DO** keep the BPE implementation from scratch and byte-level; keep it ~50
  readable lines in `src/bpe/train.py`.
- **DO** report comparisons against a *named* reference tokenizer.
- **DO** keep every quantitative claim backed by a test or a reproducible figure.
- **DON'T** add a GPU dependency, use pip, or pull a heavyweight model — CPU only.
- **DON'T** let `tiktoken`/`tokenizers` leak into the implementation path; they are
  validation only.
- **DON'T** make a notebook exceed the 3-minute CPU budget.

## Related projects (the `_101` series by Dima806)

Tokenization is the step *before* both the transformer and the embedding:

- [transformers_101](https://github.com/Dima806/transformers_101) — the architecture that consumes the token sequence (attention comes after tokens).
- [embeddings_101](https://github.com/Dima806/embeddings_101) — each token becomes a vector; tokenization decides what gets one.
- [information_theory_101](https://github.com/Dima806/information_theory_101) — BPE is compression; tokens-per-character is an information-theory measure.
- [hashing_101](https://github.com/Dima806/hashing_101) — the vocabulary as a learned dictionary over byte sequences.

## Token rules

- **Caveman mode: ON** (see `.claude/skills/caveman.md`). Code-first; terse prose.
- Run `/compact` at notebook boundaries to keep context lean.
