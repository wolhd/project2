"""Time-based interpolation of position tracks.

Operates purely in ECEF (linear interpolation in cartesian space is a
reasonable local approximation and avoids lat/lon wraparound issues; if a
future requirement needs great-circle-aware interpolation, do it on the
geodetic representation before the transform stage instead of here).

This stage buffers positions per track_id and, on each new sample, emits any
interpolated points that fall on the configured time grid between the
previous and current sample. It is intentionally stateful and per-track so
it can run indefinitely on a streaming source.
"""

from __future__ import annotations

from collections.abc import Iterator

from trackerapp.models.types import EcefPosition


class Interpolator:
    """Resamples/interpolates a per-track stream of EcefPosition onto a fixed grid."""

    def __init__(self, step_s: float, max_gap_s: float, method: str = "linear") -> None:
        if method not in ("linear", "none"):
            raise ValueError(f"unsupported interpolation method: {method}")
        self._step_s = step_s
        self._max_gap_s = max_gap_s
        self._method = method
        self._last_by_track: dict[str, EcefPosition] = {}

    def push(self, position: EcefPosition) -> Iterator[EcefPosition]:
        """Feed one new position; yields zero or more positions in time order.

        Always yields the input position last (after any interpolated
        in-between points), preserving causal order for downstream stages.
        """
        prev = self._last_by_track.get(position.track_id)
        self._last_by_track[position.track_id] = position

        if self._method == "none" or prev is None:
            yield position
            return

        gap = position.timestamp - prev.timestamp
        if gap <= 0 or gap > self._max_gap_s:
            # out-of-order, duplicate, or too large a gap to bridge safely
            yield position
            return

        yield from self._interpolate_between(prev, position)
        yield position

    def _interpolate_between(
        self, start: EcefPosition, end: EcefPosition
    ) -> Iterator[EcefPosition]:
        t = start.timestamp + self._step_s
        while t < end.timestamp:
            frac = (t - start.timestamp) / (end.timestamp - start.timestamp)
            yield EcefPosition(
                track_id=start.track_id,
                x_m=_lerp(start.x_m, end.x_m, frac),
                y_m=_lerp(start.y_m, end.y_m, frac),
                z_m=_lerp(start.z_m, end.z_m, frac),
                timestamp=t,
            )
            t += self._step_s


def _lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac
