"""Confidential-compute local proxy for E2E-encrypted model endpoints."""

__version__ = "0.1.0"

from .session import AttestationError, ConfidentialSession

__all__ = ["ConfidentialSession", "AttestationError", "__version__"]
