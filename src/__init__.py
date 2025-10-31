"""Temporal Eigenstate Networks (TEN)

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - Internal Use Only

A novel neural architecture that performs attention in eigenspace.

Author: Oluwatosin Afolabi
Company: Genovo Technologies
"""

from .model import (
    TemporalEigenstateNetwork,
    TemporalEigenstateConfig,
    EigenstateAttention,
)

__version__ = "0.1.0"
__author__ = "Oluwatosin Afolabi"
__copyright__ = "Copyright (c) 2025 Genovo Technologies. All Rights Reserved."
__license__ = "Proprietary"

__all__ = [
    "TemporalEigenstateNetwork",
    "TemporalEigenstateConfig",
    "EigenstateAttention",
]
