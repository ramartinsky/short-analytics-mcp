"""Shared validation and JSON-safety helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

MAX_ROWS = 10_000
MAX_CATEGORIES = 200
MAX_MCMC_SAMPLES = 300

Label: TypeAlias = str | int | bool


class StrictModel(BaseModel):
    """Base model used for public schemas."""

    model_config = ConfigDict(extra="forbid")


class Response(StrictModel):
    """Shared result envelope returned by every tool."""

    result: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def require_equal_lengths(**arrays: Sequence[Any]) -> int:
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Array lengths must match ({rendered}).")
    return next(iter(lengths.values()), 0)


def finite_floats(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers.")
    return array


def label_key(value: Label) -> tuple[str, str]:
    """Keep bool, int, and string labels distinct."""

    return type(value).__name__, str(value)


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite values and numpy scalars for strict JSON."""

    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def metric(value: float | None, reason: str | None = None) -> dict[str, Any]:
    return {"value": json_safe(value), "reason": reason}

