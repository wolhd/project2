"""Immutable, validated containers for every input the localizer consumes.

Conventions (SI units, radians internally)
------------------------------------------
* ``time_s``           seconds on one common epoch for *all* inputs
                       (e.g. POSIX seconds or seconds since scenario start).
* ``range_m``          EXCESS bistatic range:
                       ``|tx-tgt| + |tgt-rx| - |tx-rx|``.
* ``azimuth_rad``      direction of the tx->target vector, measured clockwise
                       from the tx->rx line of sight, in the transmitter's local
                       horizontal plane (viewed from above).
* ``range_rate_mps``   time derivative of the excess range ``range_m``.
* ``*_ecef_*``         Earth-centred Earth-fixed, WGS-84, metres / m/s.

Each single-measurement class validates and normalises its fields in
``__post_init__`` and is frozen, so an instance that exists is a valid one.
Each ``*Sequence`` class is a read-only, time-sorted ``Sequence`` that also
exposes vectorised NumPy views for estimator code.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Protocol, Self, TypeAlias, TypeVar, overload

import numpy as np

from .validation import (
    FloatArray,
    as_float,
    as_optional_positive_float,
    as_optional_vec3,
    as_positive_float,
    as_vec3,
)

_set = object.__setattr__  # the only way to normalise fields on a frozen dataclass


# --------------------------------------------------------------------------- #
# Single measurements
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True, eq=False)
class BistaticDetection:
    """One bistatic detection plus the receiver state at detection time."""

    time_s: float
    range_m: float
    azimuth_rad: float
    range_rate_mps: float
    range_std_m: float
    azimuth_std_rad: float
    range_rate_std_mps: float
    rx_pos_ecef_m: FloatArray
    rx_vel_ecef_mps: FloatArray

    def __post_init__(self) -> None:
        for name in ("time_s", "range_m", "azimuth_rad", "range_rate_mps"):
            _set(self, name, as_float(getattr(self, name), name))
        for name in ("range_std_m", "azimuth_std_rad", "range_rate_std_mps"):
            _set(self, name, as_positive_float(getattr(self, name), name))
        _set(self, "rx_pos_ecef_m", as_vec3(self.rx_pos_ecef_m, "rx_pos_ecef_m"))
        _set(self, "rx_vel_ecef_mps", as_vec3(self.rx_vel_ecef_mps, "rx_vel_ecef_mps"))


@dataclass(frozen=True, slots=True, eq=False)
class KnownAircraftState:
    """ADS-B (or otherwise known) aircraft state; velocity is optional."""

    time_s: float
    icao: str
    pos_ecef_m: FloatArray
    vel_ecef_mps: FloatArray | None = None

    def __post_init__(self) -> None:
        _set(self, "time_s", as_float(self.time_s, "time_s"))
        if not isinstance(self.icao, str) or not self.icao.strip():
            raise TypeError("icao must be a non-empty str")
        _set(self, "icao", self.icao.strip().upper())
        _set(self, "pos_ecef_m", as_vec3(self.pos_ecef_m, "pos_ecef_m"))
        _set(self, "vel_ecef_mps", as_optional_vec3(self.vel_ecef_mps, "vel_ecef_mps"))


@dataclass(frozen=True, slots=True, eq=False)
class TxLocationHint:
    """A suggested transmitter position (e.g. last known / start location)."""

    time_s: float
    pos_ecef_m: FloatArray
    pos_std_m: float | None = None  # 1-sigma radius of the suggestion, if known

    def __post_init__(self) -> None:
        _set(self, "time_s", as_float(self.time_s, "time_s"))
        _set(self, "pos_ecef_m", as_vec3(self.pos_ecef_m, "pos_ecef_m"))
        _set(self, "pos_std_m", as_optional_positive_float(self.pos_std_m, "pos_std_m"))


Measurement: TypeAlias = BistaticDetection | KnownAircraftState | TxLocationHint


class MeasurementKind(enum.IntEnum):
    """Also the tie-break order for equal timestamps: context before detections."""

    TX_HINT = 0
    KNOWN_AIRCRAFT = 1
    DETECTION = 2


def kind_of(measurement: Measurement) -> MeasurementKind:
    if isinstance(measurement, BistaticDetection):
        return MeasurementKind.DETECTION
    if isinstance(measurement, KnownAircraftState):
        return MeasurementKind.KNOWN_AIRCRAFT
    if isinstance(measurement, TxLocationHint):
        return MeasurementKind.TX_HINT
    raise TypeError(
        "expected BistaticDetection, KnownAircraftState or TxLocationHint, "
        f"got {type(measurement).__name__}"
    )


# --------------------------------------------------------------------------- #
# Sequences (group form)
# --------------------------------------------------------------------------- #
class _Timed(Protocol):
    @property
    def time_s(self) -> float: ...


T = TypeVar("T", bound=_Timed)


def _time_of(item: _Timed) -> float:
    return item.time_s


def _readonly(arr: FloatArray) -> FloatArray:
    arr.setflags(write=False)
    return arr


class TimedSequence(Sequence[T]):
    """Immutable, time-sorted, homogeneous sequence.

    Subclasses set ``_ITEM_TYPE``; items of any other type are rejected.
    Window queries are half-open, ``t_start <= t < t_end``, everywhere in this
    package, so adjacent windows never double count a sample.
    """

    _ITEM_TYPE: ClassVar[type[Any]]

    def __init__(self, items: Iterable[T] = (), *, sort: bool = True) -> None:
        collected = tuple(items)
        for idx, item in enumerate(collected):
            if not isinstance(item, self._ITEM_TYPE):
                raise TypeError(
                    f"{type(self).__name__}[{idx}] must be {self._ITEM_TYPE.__name__}, "
                    f"got {type(item).__name__}"
                )
        if sort:
            collected = tuple(sorted(collected, key=_time_of))  # stable
        elif any(b.time_s < a.time_s for a, b in zip(collected, collected[1:])):
            raise ValueError("items must be in non-decreasing time order when sort=False")
        self._items: tuple[T, ...] = collected

    # -- Sequence protocol ------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index: int | slice) -> T | Self:
        if isinstance(index, slice):
            return type(self)(self._items[index], sort=False)
        return self._items[index]

    def __repr__(self) -> str:
        if not self._items:
            return f"{type(self).__name__}(empty)"
        return f"{type(self).__name__}(n={len(self)}, t=[{self.start_time_s}, {self.end_time_s}])"

    # -- time queries ------------------------------------------------------ #
    @cached_property
    def times_s(self) -> FloatArray:
        return _readonly(np.fromiter((i.time_s for i in self._items), np.float64, len(self)))

    @property
    def start_time_s(self) -> float | None:
        return self._items[0].time_s if self._items else None

    @property
    def end_time_s(self) -> float | None:
        return self._items[-1].time_s if self._items else None

    def between(self, t_start_s: float, t_end_s: float) -> Self:
        """Items with ``t_start_s <= time_s < t_end_s``."""
        lo = int(np.searchsorted(self.times_s, t_start_s, side="left"))
        hi = int(np.searchsorted(self.times_s, t_end_s, side="left"))
        return type(self)(self._items[lo:hi], sort=False)

    def at_or_before(self, time_s: float) -> T | None:
        """Most recent item with ``item.time_s <= time_s`` (``None`` if none)."""
        idx = int(np.searchsorted(self.times_s, time_s, side="right"))
        return self._items[idx - 1] if idx else None

    # -- helpers for subclasses -------------------------------------------- #
    def _column(self, getter: Any) -> FloatArray:
        return _readonly(np.fromiter((getter(i) for i in self._items), np.float64, len(self)))

    def _rows3(self, getter: Any) -> FloatArray:
        arr = np.array([getter(i) for i in self._items], dtype=np.float64).reshape(len(self), 3)
        return _readonly(arr)


class DetectionSequence(TimedSequence[BistaticDetection]):
    _ITEM_TYPE = BistaticDetection

    @cached_property
    def ranges_m(self) -> FloatArray:
        return self._column(lambda d: d.range_m)

    @cached_property
    def azimuths_rad(self) -> FloatArray:
        return self._column(lambda d: d.azimuth_rad)

    @cached_property
    def range_rates_mps(self) -> FloatArray:
        return self._column(lambda d: d.range_rate_mps)

    @cached_property
    def range_stds_m(self) -> FloatArray:
        return self._column(lambda d: d.range_std_m)

    @cached_property
    def azimuth_stds_rad(self) -> FloatArray:
        return self._column(lambda d: d.azimuth_std_rad)

    @cached_property
    def range_rate_stds_mps(self) -> FloatArray:
        return self._column(lambda d: d.range_rate_std_mps)

    @cached_property
    def rx_pos_ecef_m(self) -> FloatArray:
        """(N, 3) receiver positions."""
        return self._rows3(lambda d: d.rx_pos_ecef_m)

    @cached_property
    def rx_vel_ecef_mps(self) -> FloatArray:
        """(N, 3) receiver velocities."""
        return self._rows3(lambda d: d.rx_vel_ecef_mps)


class KnownAircraftSequence(TimedSequence[KnownAircraftState]):
    _ITEM_TYPE = KnownAircraftState

    @cached_property
    def icaos(self) -> frozenset[str]:
        return frozenset(s.icao for s in self._items)

    @cached_property
    def pos_ecef_m(self) -> FloatArray:
        """(N, 3) positions."""
        return self._rows3(lambda s: s.pos_ecef_m)

    @cached_property
    def vel_ecef_mps(self) -> FloatArray:
        """(N, 3) velocities; rows are NaN where velocity is unknown."""
        nan3 = np.full(3, np.nan)
        return self._rows3(lambda s: nan3 if s.vel_ecef_mps is None else s.vel_ecef_mps)

    def by_icao(self) -> dict[str, KnownAircraftSequence]:
        """Split into one time-sorted track per aircraft."""
        groups: dict[str, list[KnownAircraftState]] = {}
        for state in self._items:
            groups.setdefault(state.icao, []).append(state)
        return {k: KnownAircraftSequence(v, sort=False) for k, v in sorted(groups.items())}


class TxHintSequence(TimedSequence[TxLocationHint]):
    _ITEM_TYPE = TxLocationHint

    @cached_property
    def pos_ecef_m(self) -> FloatArray:
        """(N, 3) hinted positions."""
        return self._rows3(lambda h: h.pos_ecef_m)


# --------------------------------------------------------------------------- #
# One processing group
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class MeasurementBatch:
    """Everything the estimator gets for one processing step."""

    detections: DetectionSequence = field(default_factory=DetectionSequence)
    known_aircraft: KnownAircraftSequence = field(default_factory=KnownAircraftSequence)
    tx_hints: TxHintSequence = field(default_factory=TxHintSequence)
    t_start_s: float | None = None  # nominal window, if the batch came from one
    t_end_s: float | None = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("detections", DetectionSequence),
            ("known_aircraft", KnownAircraftSequence),
            ("tx_hints", TxHintSequence),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        for name in ("t_start_s", "t_end_s"):
            value = getattr(self, name)
            if value is not None:
                _set(self, name, as_float(value, name))

    def __len__(self) -> int:
        return len(self.detections) + len(self.known_aircraft) + len(self.tx_hints)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @classmethod
    def from_measurements(
        cls,
        measurements: Iterable[Measurement],
        *,
        t_start_s: float | None = None,
        t_end_s: float | None = None,
    ) -> MeasurementBatch:
        dets: list[BistaticDetection] = []
        acs: list[KnownAircraftState] = []
        hints: list[TxLocationHint] = []
        for m in measurements:
            kind = kind_of(m)  # raises TypeError for anything else
            if kind is MeasurementKind.DETECTION:
                dets.append(m)  # type: ignore[arg-type]
            elif kind is MeasurementKind.KNOWN_AIRCRAFT:
                acs.append(m)  # type: ignore[arg-type]
            else:
                hints.append(m)  # type: ignore[arg-type]
        return cls(
            detections=DetectionSequence(dets),
            known_aircraft=KnownAircraftSequence(acs),
            tx_hints=TxHintSequence(hints),
            t_start_s=t_start_s,
            t_end_s=t_end_s,
        )
