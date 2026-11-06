"""tokenization_101 — byte-pair encoding from scratch, on CPU.

The model sees TOKENS, not characters. This package builds the BPE algorithm
behind GPT from scratch and provides probes that turn tokenization into an
explanation for why models cannot spell, why arithmetic wobbles, and why some
languages cost 2-3x the tokens (and money) for the same meaning.
"""

__version__ = "0.1.0"
