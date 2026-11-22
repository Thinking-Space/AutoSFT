"""Public consumer surface: Data Source bootstrap.

Selection and feedback (``pool_view``/``record_usage``)
live in ``selection``; this module only owns creating/repairing the durable
Data Source directory for callers such as the run launcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autosft.synthesis._internal.storage import DataSource as _DataSource


def ensure_data_source(root: str | Path) -> dict[str, Any]:
    return _DataSource(root).ensure()


__all__ = ["ensure_data_source"]
