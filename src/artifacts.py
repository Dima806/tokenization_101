"""Save notebook artifacts to disk in a consistent place.

Every figure a notebook draws is written to ``outputs/figures/*.png`` and every
textual/numerical input or result is written to ``outputs/data/*.json`` — so the
whole project reproduces to a folder of assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = _ROOT / "outputs" / "figures"
DATA_DIR = _ROOT / "outputs" / "data"


def _to_serialisable(obj: Any) -> Any:
    """Fallback encoder for JSON: dataclasses -> dict, bytes -> text."""
    if hasattr(obj, "__dataclass_fields__"):
        return vars(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def save_json(name: str, data: Any) -> Path:
    """Write ``data`` as pretty UTF-8 JSON under ``outputs/data/``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / (name if name.endswith(".json") else f"{name}.json")
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_to_serialisable),
        encoding="utf-8",
    )
    return path


def save_figure(fig: Any, name: str) -> Path:
    """Write a matplotlib figure as a PNG under ``outputs/figures/``."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
