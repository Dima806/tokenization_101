"""Shared plotting helpers for the notebooks (headless, saves to outputs/figures/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.bpe import BPETokenizer
from src.evaluation.compression import CompressionRow
from src.probes.multilingual import LanguageTax

# No explicit backend: matplotlib auto-selects Agg when there is no display
# (headless scripts/tests) and the inline backend inside notebooks.

FIGURE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def _save(fig: plt.Figure, name: str | None) -> None:
    if name:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")


def plot_language_tax(taxes: list[LanguageTax], *, save: str | None = None) -> plt.Figure:
    """Horizontal bar chart of token counts per language."""
    ordered = sorted(taxes, key=lambda t: t.num_tokens)
    labels = [t.language for t in ordered]
    values = [t.num_tokens for t in ordered]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color="#4C78A8")
    ax.set_xlabel("tokens for the same sentence")
    ax.set_title("The language tax: same meaning, different token cost")
    ax.set_xlim(0, max(values) * 1.12 if values else 1)  # headroom for value labels
    for y, value in enumerate(values):
        ax.text(value, y, f" {value}", va="center")
    _save(fig, save)
    return fig


def plot_compression(rows: list[CompressionRow], *, save: str | None = None) -> plt.Figure:
    """Bar chart of tokens-per-character across texts."""
    ordered = sorted(rows, key=lambda r: r.tokens_per_char)
    labels = [r.label for r in ordered]
    values = [r.tokens_per_char for r in ordered]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color="#72B7B2")
    ax.set_xlabel("tokens per character (lower = better compression)")
    ax.set_title("Compression view")
    _save(fig, save)
    return fig


def plot_token_breakdown(
    tokenizer: BPETokenizer, text: str, *, save: str | None = None
) -> plt.Figure:
    """Render ``text`` as its tokens, each piece in an alternating colour box."""
    pieces = tokenizer.pieces(text)
    fig, ax = plt.subplots(figsize=(min(12, 1 + 0.6 * len(pieces)), 1.6))
    ax.axis("off")
    palette = ["#4C78A8", "#F58518"]
    x = 0.0
    for index, piece in enumerate(pieces):
        width = max(len(piece), 1)
        ax.add_patch(plt.Rectangle((x, 0), width, 1, color=palette[index % 2], alpha=0.5))
        ax.text(x + width / 2, 0.5, piece.replace(" ", "·"), ha="center", va="center")
        x += width
    ax.set_xlim(0, x)
    ax.set_ylim(0, 1)
    ax.set_title(f"{len(pieces)} tokens for {len(text)} characters")
    _save(fig, save)
    return fig
