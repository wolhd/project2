"""Position/velocity states and interpolation of known (ADS-B) aircraft tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import BistaticDetection, KnownAircraftSequence
from .validation import FloatArray, as_float, as_positive_float, as_vec3

_set = object.__setattr__


@dataclass(frozen=True, slots=True, eq=False)
class KinematicState:
    """ECEF position [m] and velocity [m/s] of the transmitter, a target or the receiver."""

    pos_ecef_m: FloatArray
    vel_ecef_mps: FloatArray

    def __post_init__(self) -> None:
        _set(self, "pos_ecef_m", as_vec3(self.pos_ecef_m, "pos_ecef_m"))
        _set(self, "vel_ecef_mps", as_vec3(self.vel_ecef_mps, "vel_ecef_mps"))

    @classmethod
    def from_vector(cls, x: object) -> KinematicState:
        """Build from a length-6 ``[px, py, pz, vx, vy, vz]`` vector (estimator state)."""
        arr = np.array(x, dtype=np.float64)
        if arr.shape != (6,):
            raise ValueError(f"state vector must have shape (6,), got {arr.shape}")
        return cls(arr[:3], arr[3:])

    @classmethod
    def from_detection_receiver(cls, detection: BistaticDetection) -> KinematicState:
        if not isinstance(detection, BistaticDetection):
            raise TypeError(f"expected BistaticDetection, got {type(detection).__name__}")
        return cls(detection.rx_pos_ecef_m, detection.rx_vel_ecef_mps)

    def to_vector(self) -> FloatArray:
        return np.concatenate((self.pos_ecef_m, self.vel_ecef_mps))

    def propagate(self, dt_s: float) -> KinematicState:
        """Constant-velocity propagation by ``dt_s`` (may be negative)."""
        dt = as_float(dt_s, "dt_s")
        return KinematicState(self.pos_ecef_m + dt * self.vel_ecef_mps, self.vel_ecef_mps)


class KnownAircraftTrack:
    """Time-interpolated state of one known aircraft.

    Inside the data span: when both bracketing samples have a velocity, position
    is a cubic Hermite and velocity a linear blend of the two reported velocities;
    otherwise both are linear (velocity = finite difference).  Outside: constant-velocity extrapolation, at
    most ``max_extrapolation_s`` past either end.  Across a sampling gap longer
    than ``max_gap_s`` nothing is interpolated (``state_at`` returns ``None``),
    because ADS-B dropouts make long interpolations untrustworthy.
    """

    def __init__(
        self,
        states: KnownAircraftSequence,
        *,
        max_gap_s: float = 10.0,
        max_extrapolation_s: float = 5.0,
    ) -> None:
        if not isinstance(states, KnownAircraftSequence):
            raise TypeError(f"states must be KnownAircraftSequence, got {type(states).__name__}")
        if len(states) == 0:
            raise ValueError("a track needs at least one state")
        if len(states.icaos) != 1:
            raise ValueError(f"track must contain exactly one aircraft, got {sorted(states.icaos)}")
        self.icao: str = next(iter(states.icaos))
        self.max_gap_s = as_positive_float(max_gap_s, "max_gap_s")
        self.max_extrapolation_s = as_float(max_extrapolation_s, "max_extrapolation_s")
        if self.max_extrapolation_s < 0:
            raise ValueError("max_extrapolation_s must be >= 0")
        self._t = states.times_s
        self._p = states.pos_ecef_m
        self._v = states.vel_ecef_mps  # NaN rows = unknown

    @property
    def start_time_s(self) -> float:
        return float(self._t[0])

    @property
    def end_time_s(self) -> float:
        return float(self._t[-1])

    def _edge_velocity(self, at_end: bool) -> FloatArray | None:
        idx = -1 if at_end else 0
        if np.all(np.isfinite(self._v[idx])):
            return np.array(self._v[idx], dtype=np.float64)
        other = -2 if at_end else 1
        if len(self._t) >= 2 and self._t[idx] != self._t[other]:
            diff: FloatArray = (self._p[idx] - self._p[other]) / (self._t[idx] - self._t[other])
            return diff
        return None

    def state_at(self, time_s: float) -> KinematicState | None:
        """State at ``time_s``, or ``None`` if the track cannot support that time."""
        t = as_float(time_s, "time_s")
        n = len(self._t)
        if t < self._t[0] or t >= self._t[-1]:
            at_end = t >= self._t[-1]
            idx = -1 if at_end else 0
            dt = t - float(self._t[idx])
            if abs(dt) > self.max_extrapolation_s:
                return None
            vel = self._edge_velocity(at_end)
            if vel is None:
                return None
            return KinematicState(self._p[idx] + dt * vel, vel)

        i = int(np.searchsorted(self._t, t, side="right")) - 1  # t_i <= t < t_{i+1}
        if i + 1 >= n:  # pragma: no cover - covered by the branch above
            return None
        t0, t1 = float(self._t[i]), float(self._t[i + 1])
        h = t1 - t0
        if h > self.max_gap_s:
            return None
        p0, p1, v0, v1 = self._p[i], self._p[i + 1], self._v[i], self._v[i + 1]
        s = (t - t0) / h
        if np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)):
            h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
            pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
            # Velocity is interpolated from the *reported* velocities.  Differentiating the
            # position spline instead would amplify ADS-B position noise into velocity error.
            vel = (1.0 - s) * v0 + s * v1
        else:
            pos = p0 + s * (p1 - p0)
            vel = (p1 - p0) / h
        return KinematicState(pos, vel)


class KnownAircraftTracks:
    """One :class:`KnownAircraftTrack` per ICAO address."""

    def __init__(self, tracks: dict[str, KnownAircraftTrack]) -> None:
        self._tracks = dict(tracks)

    @classmethod
    def from_sequence(
        cls,
        states: KnownAircraftSequence,
        *,
        max_gap_s: float = 10.0,
        max_extrapolation_s: float = 5.0,
    ) -> KnownAircraftTracks:
        return cls(
            {
                icao: KnownAircraftTrack(
                    seq, max_gap_s=max_gap_s, max_extrapolation_s=max_extrapolation_s
                )
                for icao, seq in states.by_icao().items()
            }
        )

    @property
    def icaos(self) -> tuple[str, ...]:
        return tuple(self._tracks)

    def __getitem__(self, icao: str) -> KnownAircraftTrack:
        return self._tracks[icao]

    def __len__(self) -> int:
        return len(self._tracks)

    def states_at(self, time_s: float) -> dict[str, KinematicState]:
        """States of every aircraft whose track supports ``time_s``."""
        out: dict[str, KinematicState] = {}
        for icao, track in self._tracks.items():
            state = track.state_at(time_s)
            if state is not None:
                out[icao] = state
        return out
