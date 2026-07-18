"""Shared file I/O helpers used across pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def validate_input_file(csv_path: str) -> None:
    """Raise FileNotFoundError if csv_path does not exist."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Atomically save a DataFrame as UTF-8-BOM CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    df.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically save a JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
