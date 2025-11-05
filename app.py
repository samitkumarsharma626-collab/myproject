"""Compatibility module exposing the FastAPI app instance."""

from app import app  # noqa: F401  # Import for ASGI servers expecting ``app`` at top level
