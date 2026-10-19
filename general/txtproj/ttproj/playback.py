"""Time-ordered replay of the three input streams."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Protocol, TypeVar

import numpy as np

from .csv_io import StrPath, read_detections_csv, read_known_aircraft_csv, read_tx_hint_csv
from .models import (
    DetectionSequence,
    KnownAircraftSequence,
    Measurement,
    MeasurementBatch,
    TxHintSequence,
    kind_of,
)
from .validation import FloatArray, as_float, as_positive_float

_S = TypeVar("_S")


class MeasurementSink(Protocol):
    """Anything with ``add(measurement)``; :class:`~txloc.buffer.MeasurementBuffer` fits."""

    def add(self, measurement: Measurement) -> None: ...


def _checked(value: _S | None, cls: type[_S], name: str) -> _S:
    if value is None:
        return cls()  # empty sequence
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be {cls.__name__}, got {type(value).__name__}")
    return value


class MeasurementPlayback:
    """Merge detections, known-aircraft states and tx hints into one timeline.

    Ordering is by ``time_s``; ties break as tx hint -> known aircraft ->
    detection (context first), then by original order.  The merged timeline is
    built once; a cursor then walks it.  ``iter(playback)`` always replays from
    the start and does not touch the cursor.
    """

    def __init__(
        self,
        *,
        detections: DetectionSequence | None = None,
        known_aircraft: KnownAircraftSequence | None = None,
        tx_hints: TxHintSequence | None = None,
    ) -> None:
        dets = _checked(detections, DetectionSequence, "detections")
        acs = _checked(known_aircraft, KnownAircraftSequence, "known_aircraft")
        hints = _checked(tx_hints, TxHintSequence, "tx_hints")
        merged: list[Measurement] = [*dets, *acs, *hints]
        merged.sort(key=lambda m: (m.time_s, kind_of(m)))  # stable
        self._events: tuple[Measurement, ...] = tuple(merged)
        self._times: FloatArray = np.fromiter((m.time_s for m in merged), np.float64, len(merged))
        self._pos = 0

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_csv(
        cls,
        *,
        detections_path: StrPath | None = None,
        known_aircraft_path: StrPath | None = None,
        tx_hint_path: StrPath | None = None,
    ) -> MeasurementPlayback:
        return cls(
            detections=None if detections_path is None else read_detections_csv(detections_path),
            known_aircraft=(
                None if known_aircraft_path is None else read_known_aircraft_csv(known_aircraft_path)
            ),
            tx_hints=None if tx_hint_path is None else read_tx_hint_csv(tx_hint_path),
        )

    @classmethod
    def from_measurements(cls, measurements: Iterable[Measurement]) -> MeasurementPlayback:
        batch = MeasurementBatch.from_measurements(measurements)
        return cls(
            detections=batch.detections,
            known_aircraft=batch.known_aircraft,
            tx_hints=batch.tx_hints,
        )

    # -- inspection -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self._events)

    @property
    def start_time_s(self) -> float | None:
        return float(self._times[0]) if len(self._events) else None

    @property
    def end_time_s(self) -> float | None:
        return float(self._times[-1]) if len(self._events) else None

    @property
    def finished(self) -> bool:
        return self._pos >= len(self._events)

    @property
    def remaining(self) -> int:
        return len(self._events) - self._pos

    def peek_time_s(self) -> float | None:
        """Time of the next unread event, or ``None`` when finished."""
        return None if self.finished else float(self._times[self._pos])

    # -- cursor control ---------------------------------------------------- #
    def reset(self) -> None:
        self._pos = 0

    def seek(self, time_s: float) -> None:
        """Move the cursor to the first event with ``event.time_s >= time_s``."""
        t = as_float(time_s, "time_s")
        self._pos = int(np.searchsorted(self._times, t, side="left"))

    def next_event(self) -> Measurement | None:
        if self.finished:
            return None
        event = self._events[self._pos]
        self._pos += 1
        return event

    def advance_until(self, t_end_s: float) -> list[Measurement]:
        """Consume and return every unread event with ``time_s < t_end_s``."""
        t = as_float(t_end_s, "t_end_s")
        stop = int(np.searchsorted(self._times, t, side="left"))
        stop = max(stop, self._pos)
        out = list(self._events[self._pos : stop])
        self._pos = stop
        return out

    # -- higher-level drivers ---------------------------------------------- #
    def batches(
        self, window_s: float, *, start_time_s: float | None = None
    ) -> Iterator[MeasurementBatch]:
        """Yield tumbling-window batches ``[t, t+window)`` from the cursor onward.

        Windows are aligned to ``start_time_s`` (default: next event time).
        Empty windows are skipped.
        """
        width = as_positive_float(window_s, "window_s")
        if start_time_s is not None:
            self.seek(start_time_s)
        nxt = self.peek_time_s()
        if nxt is None:
            return
        t0 = nxt if start_time_s is None else as_float(start_time_s, "start_time_s")
        while (nxt := self.peek_time_s()) is not None:
            if nxt >= t0 + width:  # jump over empty windows
                t0 += math.floor((nxt - t0) / width) * width
            t1 = t0 + width
            events = self.advance_until(t1)
            if events:
                yield MeasurementBatch.from_measurements(events, t_start_s=t0, t_end_s=t1)
            t0 = t1

    def stream_to(self, sink: MeasurementSink, *, until_s: float | None = None) -> int:
        """Push events into ``sink`` (online-mode simulation); returns the count."""
        count = 0
        limit = math.inf if until_s is None else as_float(until_s, "until_s")
        while (t := self.peek_time_s()) is not None and t < limit:
            event = self.next_event()
            assert event is not None
            sink.add(event)
            count += 1
        return count
