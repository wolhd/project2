"""Bistatic measurement model  h(tx, target, rx) -> [range, range rate, azimuth].

Definitions (all positions/velocities ECEF).  With  a = p_tgt - p_tx,
b = p_rx - p_tx,  d_r = p_tgt - p_rx,  unit vectors u_t = a/|a|, u_r = d_r/|d_r|,
e = -b/|b|  (rx -> tx),  and  up = WGS-84 ellipsoid normal at the transmitter:

    range        R     = |a| + |d_r| - |b|                 (excess range)
    range rate   dR/dt = u_t.(v_tgt - v_tx) + u_r.(v_tgt - v_rx) - e.(v_tx - v_rx)
    azimuth      angle of a, clockwise from b, in the transmitter's local
                 horizontal plane viewed from above:
                     az = atan2( up.(a x b),  a.b - (a.up)(b.up) )

The estimated quantity is the transmitter state x = [p_tx, v_tx] (6,).  The
target and receiver are *known* inputs (ADS-B track / receiver navigation).

Azimuth is an angle at the transmitter, so it depends on the transmitter
position (not its velocity) and contributes a Jacobian row like range does.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, NamedTuple

import numpy as np

from .geodesy import EllipsoidFrame, ellipsoid_frame, wrap_pi
from .kinematics import KinematicState
from .models import BistaticDetection
from .validation import FloatArray, as_float, as_positive_float, as_square_matrix

#: 99 % chi-square quantiles by degrees of freedom (default gate thresholds).
CHI2_99: Final[dict[int, float]] = {1: 6.635, 2: 9.210, 3: 11.345}

_MIN_DISTANCE_M: Final[float] = 1.0
_I3: Final[FloatArray] = np.eye(3)


class DegenerateGeometryError(ValueError):
    """Raised when the geometry leaves a unit vector or horizontal angle undefined."""


class MeasurementComponent(enum.Enum):
    RANGE = "range"
    RANGE_RATE = "range_rate"
    AZIMUTH = "azimuth"


_ALL_COMPONENTS: Final = (
    MeasurementComponent.RANGE,
    MeasurementComponent.RANGE_RATE,
    MeasurementComponent.AZIMUTH,
)


@dataclass(frozen=True, slots=True, eq=False)
class PredictedMeasurement:
    range_m: float
    range_rate_mps: float
    azimuth_rad: float


class _Geometry(NamedTuple):
    a: FloatArray  # tx -> target
    b: FloatArray  # tx -> rx
    u_t: FloatArray
    r_t: float
    u_r: FloatArray
    r_r: float
    e: FloatArray  # unit vector rx -> tx
    baseline_m: float
    frame: EllipsoidFrame  # local frame at the transmitter
    az_num: float
    az_den: float


class BistaticMeasurementModel:
    """Predicts bistatic measurements and supplies EKF/gating ingredients.

    ``components`` selects and orders the rows of every vector/matrix returned
    (``residual``, ``jacobian``, ``noise_covariance`` ...).
    """

    TX_STATE_DIM: ClassVar[int] = 6

    def __init__(
        self,
        *,
        components: Sequence[MeasurementComponent] = _ALL_COMPONENTS,
        extra_std: Mapping[MeasurementComponent, float] | None = None,
    ) -> None:
        comps = tuple(components)
        if not comps or not all(isinstance(c, MeasurementComponent) for c in comps):
            raise TypeError("components must be a non-empty sequence of MeasurementComponent")
        if len(set(comps)) != len(comps):
            raise ValueError("components must be unique")
        self.components: tuple[MeasurementComponent, ...] = comps
        extra = dict(extra_std or {})
        for comp, value in extra.items():
            if not isinstance(comp, MeasurementComponent):
                raise TypeError("extra_std keys must be MeasurementComponent")
            if as_float(value, f"extra_std[{comp.name}]") < 0.0:
                raise ValueError(f"extra_std[{comp.name}] must be >= 0")
        self._extra_var = self._vector(
            *(extra.get(c, 0.0) ** 2 for c in _ALL_COMPONENTS)
        )  # per-component unmodelled variance (e.g. ADS-B error), in model component order

    @property
    def dim(self) -> int:
        return len(self.components)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_states(*states: KinematicState) -> None:
        for s in states:
            if not isinstance(s, KinematicState):
                raise TypeError(f"expected KinematicState, got {type(s).__name__}")

    @staticmethod
    def _geometry(tx: KinematicState, target: KinematicState, rx: KinematicState) -> _Geometry:
        a = target.pos_ecef_m - tx.pos_ecef_m
        b = rx.pos_ecef_m - tx.pos_ecef_m
        d_r = target.pos_ecef_m - rx.pos_ecef_m
        r_t, r_r, baseline = (float(np.linalg.norm(v)) for v in (a, d_r, b))
        if min(r_t, r_r, baseline) < _MIN_DISTANCE_M:
            raise DegenerateGeometryError("target, transmitter and receiver must be distinct points")
        frame = ellipsoid_frame(tx.pos_ecef_m)
        up = frame.up
        au, bu = float(a @ up), float(b @ up)
        if (
            float(np.linalg.norm(a - au * up)) < _MIN_DISTANCE_M
            or float(np.linalg.norm(b - bu * up)) < _MIN_DISTANCE_M
        ):
            raise DegenerateGeometryError(
                "target or receiver is directly above/below the transmitter: azimuth undefined"
            )
        return _Geometry(
            a=a,
            b=b,
            u_t=a / r_t,
            r_t=r_t,
            u_r=d_r / r_r,
            r_r=r_r,
            e=-b / baseline,
            baseline_m=baseline,
            frame=frame,
            az_num=float(up @ np.cross(a, b)),
            az_den=float(a @ b - au * bu),
        )

    def predict(
        self, tx: KinematicState, target: KinematicState, rx: KinematicState
    ) -> PredictedMeasurement:
        """Noise-free measurement for the given geometry (azimuth in [-pi, pi])."""
        self._check_states(tx, target, rx)
        g = self._geometry(tx, target, rx)
        rng = g.r_t + g.r_r - g.baseline_m
        rate = float(
            g.u_t @ (target.vel_ecef_mps - tx.vel_ecef_mps)
            + g.u_r @ (target.vel_ecef_mps - rx.vel_ecef_mps)
            - g.e @ (tx.vel_ecef_mps - rx.vel_ecef_mps)
        )
        return PredictedMeasurement(
            range_m=rng, range_rate_mps=rate, azimuth_rad=math.atan2(g.az_num, g.az_den)
        )

    def jacobian(
        self, tx: KinematicState, target: KinematicState, rx: KinematicState
    ) -> FloatArray:
        """``d h / d [p_tx, v_tx]`` with shape ``(dim, 6)``; azimuth has no velocity term."""
        self._check_states(tx, target, rx)
        g = self._geometry(tx, target, rx)
        up, jn = g.frame.up, g.frame.up_jacobian

        # range: d/dp_tx = -u_t - e ; range rate: d/dv_tx is the same vector
        drdp = -g.u_t - g.e
        w_t = target.vel_ecef_mps - tx.vel_ecef_mps
        w_b = tx.vel_ecef_mps - rx.vel_ecef_mps
        ratedp = -(_I3 - np.outer(g.u_t, g.u_t)) @ w_t / g.r_t - (
            _I3 - np.outer(g.e, g.e)
        ) @ w_b / g.baseline_m

        # azimuth = atan2(num, den); gradients w.r.t. p_tx include the tilt of `up`
        au, bu = float(g.a @ up), float(g.b @ up)
        d_r = target.pos_ecef_m - rx.pos_ecef_m
        g_num = np.cross(d_r, up) + jn @ np.cross(g.a, g.b)
        g_den = -(g.a + g.b) + (au + bu) * up - jn @ (bu * g.a + au * g.b)
        azdp = (g.az_den * g_num - g.az_num * g_den) / (g.az_num**2 + g.az_den**2)

        rows: dict[MeasurementComponent, FloatArray] = {
            MeasurementComponent.RANGE: np.concatenate((drdp, np.zeros(3))),
            MeasurementComponent.RANGE_RATE: np.concatenate((ratedp, drdp)),
            MeasurementComponent.AZIMUTH: np.concatenate((azdp, np.zeros(3))),
        }
        return np.vstack([rows[c] for c in self.components])

    # ------------------------------------------------------------------ #
    def _vector(self, rng: float, rate: float, az: float) -> FloatArray:
        values = {
            MeasurementComponent.RANGE: rng,
            MeasurementComponent.RANGE_RATE: rate,
            MeasurementComponent.AZIMUTH: az,
        }
        return np.array([values[c] for c in self.components], dtype=np.float64)

    def residual(
        self, detection: BistaticDetection, tx: KinematicState, target: KinematicState
    ) -> FloatArray:
        """``measured - predicted`` (azimuth wrapped to [-pi, pi)), shape ``(dim,)``."""
        if not isinstance(detection, BistaticDetection):
            raise TypeError(f"expected BistaticDetection, got {type(detection).__name__}")
        pred = self.predict(tx, target, KinematicState.from_detection_receiver(detection))
        res = self._vector(
            detection.range_m - pred.range_m,
            detection.range_rate_mps - pred.range_rate_mps,
            detection.azimuth_rad - pred.azimuth_rad,
        )
        if MeasurementComponent.AZIMUTH in self.components:
            i = self.components.index(MeasurementComponent.AZIMUTH)
            res[i] = wrap_pi(res[i])
        return res

    def noise_covariance(self, detection: BistaticDetection) -> FloatArray:
        """Diagonal ``R`` (detection sigmas plus any ``extra_std``), shape ``(dim, dim)``."""
        var = self._vector(
            detection.range_std_m**2,
            detection.range_rate_std_mps**2,
            detection.azimuth_std_rad**2,
        )
        return np.diag(var + self._extra_var)

    def innovation_covariance(
        self,
        detection: BistaticDetection,
        tx: KinematicState,
        target: KinematicState,
        tx_cov: object,
    ) -> FloatArray:
        """``S = H P H^T + R`` for a 6x6 transmitter-state covariance ``P``."""
        p = as_square_matrix(tx_cov, self.TX_STATE_DIM, "tx_cov")
        h = self.jacobian(tx, target, KinematicState.from_detection_receiver(detection))
        return np.asarray(h @ p @ h.T + self.noise_covariance(detection), dtype=np.float64)

    def mahalanobis_sq(
        self,
        detection: BistaticDetection,
        tx: KinematicState,
        target: KinematicState,
        tx_cov: object,
    ) -> float:
        y = self.residual(detection, tx, target)
        s = self.innovation_covariance(detection, tx, target, tx_cov)
        return float(y @ np.linalg.solve(s, y))

    def gate(
        self,
        detection: BistaticDetection,
        tx: KinematicState,
        target: KinematicState,
        tx_cov: object,
        *,
        chi2_threshold: float | None = None,
    ) -> bool:
        """True if the detection is statistically consistent with (tx, target)."""
        limit = (
            CHI2_99[self.dim]
            if chi2_threshold is None
            else as_positive_float(chi2_threshold, "chi2_threshold")
        )
        return self.mahalanobis_sq(detection, tx, target, tx_cov) <= limit

    # ------------------------------------------------------------------ #
    def simulate_detection(
        self,
        time_s: float,
        tx: KinematicState,
        target: KinematicState,
        rx: KinematicState,
        *,
        range_std_m: float,
        azimuth_std_rad: float,
        range_rate_std_mps: float,
        rng: np.random.Generator | None = None,
    ) -> BistaticDetection:
        """Synthesise a detection; noise is added only when ``rng`` is given."""
        pred = self.predict(tx, target, rx)
        noise = np.zeros(3)
        if rng is not None:
            noise = rng.normal(0.0, [range_std_m, range_rate_std_mps, azimuth_std_rad])
        return BistaticDetection(
            time_s=time_s,
            range_m=pred.range_m + float(noise[0]),
            azimuth_rad=pred.azimuth_rad + float(noise[2]),
            range_rate_mps=pred.range_rate_mps + float(noise[1]),
            range_std_m=range_std_m,
            azimuth_std_rad=azimuth_std_rad,
            range_rate_std_mps=range_rate_std_mps,
            rx_pos_ecef_m=rx.pos_ecef_m,
            rx_vel_ecef_mps=rx.vel_ecef_mps,
        )
