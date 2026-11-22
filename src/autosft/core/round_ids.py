"""Strict round identifiers used by the current AutoSFT artifact contract."""

from __future__ import annotations

import re
from typing import Any


LOGICAL_ROUND_RE = re.compile(r"^[1-9][0-9]*$")
STORAGE_ROUND_RE = re.compile(r"^round_([1-9][0-9]*)$")

# Rounds are numbered from 1. Round 1 is the bootstrap the meta agent authors from
# the staged pool before any analysis exists.
FIRST_ROUND_ID = "1"


class RoundIdError(ValueError):
    pass


def logical_round_id(
    value: Any,
    *,
    label: str = "round_id",
) -> str:
    """Return an exact logical round id, normalizing JSON integers.

    Context files are JSON, so an orchestrator naturally writes ``9`` where
    the artifact contract stores ``"9"``. Both name the same round; accept
    the integer form and canonicalize to the string form instead of failing
    the round on a type mismatch.
    """

    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        value = str(value)
    if not isinstance(value, str) or LOGICAL_ROUND_RE.fullmatch(value) is None:
        raise RoundIdError(
            f"{label} must be a positive integer (JSON number or string) "
            f'without leading zeros, e.g. 9 or "9"; got {value!r}'
        )
    return value


def storage_round_id(value: Any, *, label: str = "round storage id") -> str:
    """Return an exact ``round_<n>`` storage id for a positive round."""

    if not isinstance(value, str) or STORAGE_ROUND_RE.fullmatch(value) is None:
        raise RoundIdError(
            f"{label} must match 'round_[1-9][0-9]*' without aliases or leading zeros"
        )
    return value


def logical_id_from_storage(value: Any, *, label: str = "round storage id") -> str:
    canonical = storage_round_id(value, label=label)
    match = STORAGE_ROUND_RE.fullmatch(canonical)
    assert match is not None
    return match.group(1)


# Base of the runner-derived per-round sampling seed. Seeds are round
# infrastructure, not an experiment variable: the runner derives one from the
# round id so every round samples differently and reproducibly without any
# agent-authored seed bookkeeping.
DERIVED_SEED_BASE = 42


def derived_seed(round_id: Any) -> int:
    """Deterministic sampling seed for one round, owned by the runner."""

    return DERIVED_SEED_BASE + int(logical_round_id(round_id))
