"""Strategy implementations and lookup helpers."""

from __future__ import annotations

import inspect
from typing import Dict, Type

from . import strategy_base
from .strategy_base import (
    Strategy,
    SentimentMomentumStrategy
)


def _build_registry() -> Dict[str, Type[Strategy]]:
    registry: Dict[str, Type[Strategy]] = {}
    for name, obj in inspect.getmembers(strategy_base, inspect.isclass):
        if obj is Strategy:
            continue
        if issubclass(obj, Strategy) and obj.__module__ == strategy_base.__name__:
            registry[name.lower()] = obj
    registry.setdefault("sentiment", SentimentMomentumStrategy)
    return registry


_REGISTRY = _build_registry()


def get_strategy_class(name: str) -> Type[Strategy]:
    key = name.strip().lower()
    if not key:
        raise ValueError("Strategy name cannot be empty.")
    if key not in _REGISTRY:
        options = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {options}")
    return _REGISTRY[key]


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = [
    "Strategy",
    "SentimentMomentumStrategy",
    "get_strategy_class",
    "list_strategies",
]
