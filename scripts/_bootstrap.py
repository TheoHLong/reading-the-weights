from __future__ import annotations

from pathlib import Path
import sys


def ensure_project_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def ensure_src_on_path() -> None:
    """Backward-compatible alias for older scripts."""
    ensure_project_on_path()
