"""Public modular Synthesis interfaces; internal generation remains a black box.

Consumers browse a pinned pool view, inspect batch content and evidence, choose
the rows that test the current training hypothesis, and then call
``record_usage`` so the pool learns what it actually contributed. The API keeps
the selection judgment with the consumer while making the resulting exposure
auditable.
"""

from autosft.synthesis.api import ensure_data_source
from autosft.synthesis.selection import pool_view, record_usage

__all__ = [
    "ensure_data_source",
    "pool_view",
    "record_usage",
]
