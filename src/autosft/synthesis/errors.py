"""Exception hierarchy for the modular Synthesis control plane."""

from __future__ import annotations


class SynError(RuntimeError):
    """Base class for recoverable Synthesis control-plane failures."""


class SynConfigError(SynError):
    """Raised when runtime configuration is missing or invalid."""


class SynContractError(SynError):
    """Raised when an external or persisted Synthesis contract is invalid."""


class SynPathError(SynContractError):
    """Raised when an identifier or path escapes its declared boundary."""


class SynRequestError(SynContractError):
    """Raised when a synthesis request cannot be discovered or frozen."""


class SynPromptError(SynContractError):
    """Raised when a prompt template cannot be rendered exactly."""


class SynValidationError(SynContractError):
    """Raised when a worker delivery does not satisfy the public contract."""


class SynIOError(SynError):
    """Raised when a durable filesystem operation cannot be completed."""


class SynConflictError(SynError):
    """Raised when an idempotent write observes different existing state."""
