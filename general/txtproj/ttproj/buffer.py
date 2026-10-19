"""Accumulator that serves the estimator in online mode or file mode.

Online mode : call ``add*`` as data arrives, then either
              ``snapshot(t0, t1)``  - non-destructive sliding window, or
              ``pop_before(t1)``    - destructive tumbling window, and
              ``prune_before(t)``   - forget old data.
File mode   : ``extend(playback)`` (or ``add``) everything, then ``all()`` for
              one batch, or ``iter_windows(window, step)`` for sliding windows.

Out-of-order arrival is handled (items are inserted in time order).  All
methods are guarded by a re-entrant lock so a reader thread and a processing
thread can share one buffer.  Windows are half-open: ``t_start <= t < t_end``.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_left, insort
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

from .models import (
    BistaticDetection,
    DetectionSequence,
    KnownAircraftSequence,
    KnownAircraftState,
    Measurement,
    MeasurementBatch,
    MeasurementKind,
    TxHintSequence,
    TxLocationHint,
    _Timed,
    kind_of,
)
from .validation import as_float, as_positive_float

T = TypeVar("T", bound=_Timed)


def _time_of(item: _Timed) -> float:
    return item.time_s


class _TimeOrderedStore(Generic[T]):
    """List kept sorted by ``time_s``; equal times keep arrival order."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: T) -> None:
        insort(self._items, item, key=_time_of)

    def window(self, t_start: float, t_end: float) -> list[T]:
        lo = bisect_left(self._items, t_start, key=_time_of)
        hi = bisect_left(self._items, t_end, key=_time_of)
        return self._items[lo:hi]

    def pop_before(self, t_end: float) -> list[T]:
        hi = bisect_left(self._items, t_end, key=_time_of)
        popped = self._items[:hi]
        del self._items[:hi]
        return popped

    def clear(self) -> None:
        self._items.clear()

    @property
    def first_time(self) -> float | None:
        return self._items[0].time_s if self._items else None

    @property
    def last_time(self) -> float | None:
        return self._items[-1].time_s if self._items else None


class MeasurementBuffer:
    """Stores all three input types and hands out :class:`MeasurementBatch` groups."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._detections: _TimeOrderedStore[BistaticDetection] = _TimeOrderedStore()
        self._aircraft: _TimeOrderedStore[KnownAircraftState] = _TimeOrderedStore()
        self._hints: _TimeOrderedStore[TxLocationHint] = _TimeOrderedStore()

    # -- input ------------------------------------------------------------- #
    def add_detection(self, detection: BistaticDetection) -> None:
        if not isinstance(detection, BistaticDetection):
            raise TypeError(f"expected BistaticDetection, got {type(detection).__name__}")
        with self._lock:
            self._detections.add(detection)

    def add_known_aircraft(self, state: KnownAircraftState) -> None:
        if not isinstance(state, KnownAircraftState):
            raise TypeError(f"expected KnownAircraftState, got {type(state).__name__}")
        with self._lock:
            self._aircraft.add(state)

    def add_tx_hint(self, hint: TxLocationHint) -> None:
        if not isinstance(hint, TxLocationHint):
            raise TypeError(f"expected TxLocationHint, got {type(hint).__name__}")
        with self._lock:
            self._hints.add(hint)

    def add(self, measurement: Measurement) -> None:
        """Dispatch on type; this is also what :class:`MeasurementSink` requires."""
        kind = kind_of(measurement)  # TypeError for unsupported types
        if kind is MeasurementKind.DETECTION:
            self.add_detection(measurement)  # type: ignore[arg-type]
        elif kind is MeasurementKind.KNOWN_AIRCRAFT:
            self.add_known_aircraft(measurement)  # type: ignore[arg-type]
        else:
            self.add_tx_hint(measurement)  # type: ignore[arg-type]

    def extend(self, measurements: Iterable[Measurement]) -> int:
        """Add many items (e.g. a whole ``MeasurementPlayback``); returns the count."""
        count = 0
        with self._lock:
            for m in measurements:
                self.add(m)
                count += 1
        return count

    # -- state ------------------------------------------------------------- #
    def __len__(self) -> int:
        with self._lock:
            return len(self._detections) + len(self._aircraft) + len(self._hints)

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "detections": len(self._detections),
                "known_aircraft": len(self._aircraft),
                "tx_hints": len(self._hints),
            }

    @property
    def latest_time_s(self) -> float | None:
        """Newest timestamp held (a natural 'watermark' for online processing)."""
        with self._lock:
            times = [
                t
                for t in (self._detections.last_time, self._aircraft.last_time, self._hints.last_time)
                if t is not None
            ]
        return max(times) if times else None

    def latest_tx_hint(self, at_or_before_s: float | None = None) -> TxLocationHint | None:
        """Newest hint, optionally restricted to ``time_s <= at_or_before_s``."""
        limit = math.inf if at_or_before_s is None else as_float(at_or_before_s, "at_or_before_s")
        with self._lock:
            hints = self._hints.window(-math.inf, math.nextafter(limit, math.inf))
        return hints[-1] if hints else None

    def clear(self) -> None:
        with self._lock:
            self._detections.clear()
            self._aircraft.clear()
            self._hints.clear()

    # -- output ------------------------------------------------------------ #
    def snapshot(
        self, t_start_s: float | None = None, t_end_s: float | None = None
    ) -> MeasurementBatch:
        """Non-destructive copy of everything in ``[t_start_s, t_end_s)``; ``None`` = open."""
        lo = -math.inf if t_start_s is None else as_float(t_start_s, "t_start_s")
        hi = math.inf if t_end_s is None else as_float(t_end_s, "t_end_s")
        with self._lock:
            return MeasurementBatch(
                detections=DetectionSequence(self._detections.window(lo, hi), sort=False),
                known_aircraft=KnownAircraftSequence(self._aircraft.window(lo, hi), sort=False),
                tx_hints=TxHintSequence(self._hints.window(lo, hi), sort=False),
                t_start_s=t_start_s,
                t_end_s=t_end_s,
            )

    def all(self) -> MeasurementBatch:
        """File mode: everything stored, as a single batch."""
        return self.snapshot()

    def pop_before(self, t_end_s: float) -> MeasurementBatch:
        """Remove and return everything with ``time_s < t_end_s`` (tumbling window)."""
        hi = as_float(t_end_s, "t_end_s")
        with self._lock:
            return MeasurementBatch(
                detections=DetectionSequence(self._detections.pop_before(hi), sort=False),
                known_aircraft=KnownAircraftSequence(self._aircraft.pop_before(hi), sort=False),
                tx_hints=TxHintSequence(self._hints.pop_before(hi), sort=False),
                t_end_s=hi,
            )

    def prune_before(self, t_end_s: float) -> int:
        """Discard everything older than ``t_end_s``; returns how many items were dropped."""
        return len(self.pop_before(t_end_s))

    def iter_windows(
        self, window_s: float, step_s: float | None = None
    ) -> Iterator[MeasurementBatch]:
        """Sliding windows over the stored data (``step_s`` defaults to ``window_s``).

        Non-destructive; empty windows are skipped.  Intended for file mode.
        """
        width = as_positive_float(window_s, "window_s")
        step = width if step_s is None else as_positive_float(step_s, "step_s")
        with self._lock:
            firsts = [
                t
                for t in (self._detections.first_time, self._aircraft.first_time, self._hints.first_time)
                if t is not None
            ]
        last = self.latest_time_s
        if not firsts or last is None:
            return
        first = min(firsts)
        k = 0
        while (t0 := first + k * step) <= last:
            batch = self.snapshot(t0, t0 + width)
            if not batch.is_empty:
                yield batch
            k += 1
