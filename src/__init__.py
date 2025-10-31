"""Temporal Eigenstate Networks (TEN)

A novel neural architecture that performs attention in eigenspace.
"""

from .model import (
    TemporalEigenstateNetwork,
    TemporalEigenstateConfig,
    EigenstateAttention,
)

__version__ = "0.1.0"

__all__ = [
    "TemporalEigenstateNetwork",
    "TemporalEigenstateConfig",
    "EigenstateAttention",
]
