"""Small runtime validators used by the data classes.

Static type hints only help if a type checker is run, so the public data
classes also validate at construction time.  Everything here either returns a
clean, normalised value (``float`` or read-only ``float64`` ndarray) or raises
``TypeError`` (wrong kind of object) / ``ValueError`` (right kind, bad value).
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def as_float(value: object, name: str) -> float:
    """Return ``value`` as a finite ``float``; reject bools, strings, NaN, inf."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out}")
    return out


def as_positive_float(value: object, name: str) -> float:
    """Like :func:`as_float` but additionally requires ``> 0`` (std devs, windows)."""
    out = as_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out}")
    return out


def as_vec3(value: object, name: str) -> FloatArray:
    """Return a private, read-only ``(3,)`` float64 copy of ``value``."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a length-3 numeric sequence, got {type(value).__name__}")
    try:
        arr: FloatArray = np.array(value, dtype=np.float64)  # always copies
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a length-3 numeric sequence") from exc
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    arr.setflags(write=False)
    return arr


def as_optional_vec3(value: object | None, name: str) -> FloatArray | None:
    return None if value is None else as_vec3(value, name)


def as_optional_positive_float(value: object | None, name: str) -> float | None:
    return None if value is None else as_positive_float(value, name)
