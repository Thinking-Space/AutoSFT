"""Shared local SSTable tokenizer runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


REQUIRED_TOKENIZER_FILES = (Path("prepare_tokens.py"),)


def missing_tokenizer_files(code_dir: str | Path) -> tuple[Path, ...]:
    root = Path(code_dir).expanduser().resolve(strict=False)
    return tuple(
        root / relative
        for relative in REQUIRED_TOKENIZER_FILES
        if not (root / relative).is_file()
    )


def tokenizer_subprocess_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Keep CPU tokenization independent of optional model backend ABIs."""

    return {
        **dict(os.environ if environ is None else environ),
        "PYTHONNOUSERSITE": "1",
        "USE_TORCH": "0",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }


__all__ = [
    "REQUIRED_TOKENIZER_FILES",
    "missing_tokenizer_files",
    "tokenizer_subprocess_env",
]
