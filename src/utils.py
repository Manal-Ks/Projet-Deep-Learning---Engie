from __future__ import annotations
from pathlib import Path
import json


def ensure_dir(path):
    """
    Create directory if it doesn't exist and return Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    """
    Save dict as JSON file.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
