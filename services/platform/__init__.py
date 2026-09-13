"""Stable platform primitives shared by Ragbot runtime surfaces."""

from .registry import RegistrableComponent, TypedRegistry
from .runtime import RuntimeProfile

__all__ = ["RegistrableComponent", "RuntimeProfile", "TypedRegistry"]
