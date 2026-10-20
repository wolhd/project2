"""Batch least-squares localization of a STATIONARY transmitter.

Unknown   x = [px, py, pz]  (transmitter ECEF position, 3 numbers).

Residuals (all in units of standard deviations, stacked into one vector):
    for each detection j associated with known aircraft k(j):
        (z_j - h(x, aircraft_k at t_j, rx at t_j)) / sigma_j      -> dim values
    plus the hint as a prior:
        (x - x_hint) / sigma_hint                                  -> 3 values
``scipy.optimize.least_squares`` minimises the (robust) sum of squares using
the analytic Jacobian from the measurement model.  The covariance is
``(J^T J)^-1`` at the solution.

The hard part is not the optimiser but *association*: most detections come from
unknown aircraft or clutter.  The pipeline is therefore

  1. HYPOTHESIZE  (RANSAC-style).  One detection + one known aircraft gives 3
     measurements for 3 unknowns, so a single pair already fixes a candidate
     position (with the hint prior as regulariser).  Try many random pairs, score
     each candidate by how many *other* detections it explains, keep the best.
  2. REFINE.  Gate all detections against the current estimate and covariance,
     keep the best aircraft per detection, re-solve, repeat until the
     assignments stop changing.

Simplifying assumptions (see the project notes): static transmitter, nearest-
neighbour association (one aircraft per detection, no probabilistic
association), ADS-B error handled as a fixed extra sigma (``extra_*_std``), and a
crude fixed sigma inflation when scoring hypotheses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
from scipy.optimize import least_squares

from .association import TxPrediction, gate_candidates
from .kinematics import KinematicState, KnownAircraftTracks
from .measurement_model import (
    CHI2_99,
    BistaticMeasurementModel,
    DegenerateGeometryError,
    MeasurementComponent,
)
from .models import BistaticDetection, DetectionSequence, MeasurementBatch, TxLocationHint
from .validation import FloatArray, as_float, as_positive_float

Loss = Literal["linear", "huber", "soft_l1", "cauchy"]
_ZERO3: Final[FloatArray] = np.zeros(3)
_STATIC_VEL_VAR: Final[float] = 1e-6  # (m/s)^2: "velocity is zero" in the 6x6 gating covariance


@dataclass(frozen=True, slots=True)
class EstimatorConfig:
    components: tuple[MeasurementComponent, ...] = (
        MeasurementComponent.RANGE,
        MeasurementComponent.RANGE_RATE,
        MeasurementComponent.AZIMUTH,
    )
    default_hint_std_m: float = 20_000.0  # used when the hint has no pos_std_m
    # Error allowance for the known aircraft's ADS-B state, added in quadrature to each
    # detection's sigmas (simplification: constants instead of propagating ADS-B covariance).
    extra_range_std_m: float = 20.0
    extra_range_rate_std_mps: float = 0.5
    extra_azimuth_std_rad: float = 0.0
    loss: Loss = "huber"
    f_scale: float = 1.0  # robust-loss knee, in sigmas
    max_nfev: int = 100
    # hypothesis stage
    n_hypotheses: int = 80
    n_scoring_detections: int = 150
    score_sigma_inflation: float = 5.0  # residuals are divided by this when scoring candidates
    seed: int = 0
    # refinement stage
    max_association_rounds: int = 10
    gate_chi2: float | None = None  # default: 99 % quantile for the model dimension

    def __post_init__(self) -> None:
        if self.loss not in get_args(Loss):
            raise ValueError(f"loss must be one of {get_args(Loss)}, got {self.loss!r}")
        for name in ("default_hint_std_m", "f_scale", "score_sigma_inflation"):
            as_positive_float(getattr(self, name), name)
        for name in ("extra_range_std_m", "extra_range_rate_std_mps", "extra_azimuth_std_rad"):
            if as_float(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("max_nfev", "n_hypotheses", "n_scoring_detections", "max_association_rounds"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive int")
        if self.gate_chi2 is not None:
            as_positive_float(self.gate_chi2, "gate_chi2")


@dataclass(frozen=True, slots=True, eq=False)
class Assignment:
    detection_index: int
    icao: str
    mahalanobis_sq: float


@dataclass(frozen=True, slots=True, eq=False)
class TxEstimate:
    position_ecef_m: FloatArray  # (3,)
    covariance_m2: FloatArray  # (3, 3)
    cost: float  # 0.5 * sum of (robust) squared normalised residuals
    assignments: tuple[Assignment, ...]
    n_detections: int
    singular_values: FloatArray  # of the normalised Jacobian: small value = weak direction
    association_rounds: int
    converged: bool
    message: str

    @property
    def state(self) -> KinematicState:
        return KinematicState(self.position_ecef_m, _ZERO3)

    @property
    def position_std_m(self) -> FloatArray:
        return np.sqrt(np.diag(self.covariance_m2))


@dataclass(frozen=True, slots=True, eq=False)
class _Pair:
    """One detection tied to one known aircraft (the unit the optimiser works on)."""

    detection_index: int
    icao: str
    detection: BistaticDetection
    target: KinematicState
    rx: KinematicState
    sigma: FloatArray  # (dim,)


@dataclass(frozen=True, slots=True, eq=False)
class _Fit:
    x: FloatArray
    cov: FloatArray
    cost: float
    singular_values: FloatArray
    success: bool
    message: str


def _cov6(cov3: FloatArray) -> FloatArray:
    out = np.zeros((6, 6))
    out[:3, :3] = 0.5 * (cov3 + cov3.T)
    out[3:, 3:] = np.eye(3) * _STATIC_VEL_VAR
    return out


class StaticTxEstimator:
    def __init__(self, config: EstimatorConfig | None = None) -> None:
        self.config = config or EstimatorConfig()
        cfg = self.config
        self._model = BistaticMeasurementModel(
            components=cfg.components,
            extra_std={
                MeasurementComponent.RANGE: cfg.extra_range_std_m,
                MeasurementComponent.RANGE_RATE: cfg.extra_range_rate_std_mps,
                MeasurementComponent.AZIMUTH: cfg.extra_azimuth_std_rad,
            },
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def estimate(self, batch: MeasurementBatch) -> TxEstimate:
        """Localise the transmitter from one batch (detections + ADS-B + hint)."""
        if not isinstance(batch, MeasurementBatch):
            raise TypeError(f"batch must be MeasurementBatch, got {type(batch).__name__}")
        if len(batch.tx_hints) == 0:
            raise ValueError("a tx hint is required (it initialises and regularises the solution)")
        hint = self._select_hint(batch)
        hint_std = hint.pos_std_m or self.config.default_hint_std_m
        prior_cov = np.eye(3) * hint_std**2

        detections = batch.detections
        tracks = KnownAircraftTracks.from_sequence(batch.known_aircraft)
        if len(detections) == 0 or len(tracks) == 0:
            return self._prior_only(hint, prior_cov, len(detections), "no detections or no known aircraft")

        start = self._hypothesize(detections, tracks, hint.pos_ecef_m, hint_std)
        if start is None:
            return self._prior_only(hint, prior_cov, len(detections), "no usable hypothesis")
        x, cov = start

        previous: frozenset[tuple[int, str]] | None = None
        assignments: list[Assignment] = []
        fit: _Fit | None = None
        rounds = 0
        stable = False
        for rounds in range(1, self.config.max_association_rounds + 1):
            assignments = self._associate(detections, tracks, x, cov)
            if not assignments:
                break
            fit = self._solve(self._pairs(detections, tracks, assignments), x, hint.pos_ecef_m, hint_std)
            x, cov = fit.x, fit.cov
            key = frozenset((a.detection_index, a.icao) for a in assignments)
            if key == previous:
                stable = True
                break
            previous = key

        if fit is None:
            return TxEstimate(x, cov, float("nan"), (), len(detections), np.zeros(0), rounds, False,
                              "hypothesis found but nothing passed the gate")  # fmt: skip
        return TxEstimate(
            position_ecef_m=fit.x,
            covariance_m2=fit.cov,
            cost=fit.cost,
            assignments=tuple(assignments),
            n_detections=len(detections),
            singular_values=fit.singular_values,
            association_rounds=rounds,
            converged=stable and fit.success,
            message=fit.message if stable else "association did not stabilise: " + fit.message,
        )

    # ------------------------------------------------------------------ #
    # least squares
    # ------------------------------------------------------------------ #
    def _solve(
        self, pairs: Sequence[_Pair], x0: FloatArray, prior_pos: FloatArray, prior_std: float
    ) -> _Fit:
        model = self._model

        def residuals(x: FloatArray) -> FloatArray:
            tx = KinematicState(x, _ZERO3)
            parts = [model.residual(p.detection, tx, p.target) / p.sigma for p in pairs]
            parts.append((x - prior_pos) / prior_std)
            return np.concatenate(parts)

        def jacobian(x: FloatArray) -> FloatArray:
            tx = KinematicState(x, _ZERO3)
            rows = [-model.jacobian(tx, p.target, p.rx)[:, :3] / p.sigma[:, None] for p in pairs]
            rows.append(np.eye(3) / prior_std)
            return np.vstack(rows)

        res = least_squares(
            residuals,
            x0,
            jac=jacobian,
            loss=self.config.loss,
            f_scale=self.config.f_scale,
            x_scale=np.full(3, prior_std),
            max_nfev=self.config.max_nfev,
        )
        j = res.jac
        cov = np.linalg.pinv(j.T @ j)
        return _Fit(
            x=np.asarray(res.x, dtype=np.float64),
            cov=0.5 * (cov + cov.T),
            cost=float(res.cost),
            singular_values=np.linalg.svd(j, compute_uv=False),
            success=bool(res.success),
            message=str(res.message),
        )

    # ------------------------------------------------------------------ #
    # stage 1: hypothesize
    # ------------------------------------------------------------------ #
    def _hypothesize(
        self,
        detections: DetectionSequence,
        tracks: KnownAircraftTracks,
        hint_pos: FloatArray,
        hint_std: float,
    ) -> tuple[FloatArray, FloatArray] | None:
        cfg, model = self.config, self._model
        rng = np.random.default_rng(cfg.seed)

        # every (detection, aircraft) pair the ADS-B tracks can support
        candidates: list[_Pair] = []
        per_detection: dict[int, list[_Pair]] = {}
        for idx, det in enumerate(detections):
            rx = KinematicState.from_detection_receiver(det)
            sigma = np.sqrt(np.diag(model.noise_covariance(det)))
            for icao, target in tracks.states_at(det.time_s).items():
                pair = _Pair(idx, icao, det, target, rx, sigma)
                candidates.append(pair)
                per_detection.setdefault(idx, []).append(pair)
        if not candidates:
            return None

        scoring_ids = rng.choice(
            sorted(per_detection), size=min(cfg.n_scoring_detections, len(per_detection)), replace=False
        )
        scoring = [per_detection[int(i)] for i in scoring_ids]
        limit = CHI2_99[model.dim] if cfg.gate_chi2 is None else cfg.gate_chi2

        best: tuple[int, float, FloatArray, FloatArray] | None = None  # (inliers, -score, x, cov)
        for ci in rng.choice(len(candidates), size=min(cfg.n_hypotheses, len(candidates)), replace=False):
            try:
                fit = self._solve([candidates[int(ci)]], hint_pos, hint_pos, hint_std)
                inliers, score = self._score(fit.x, scoring, limit)
            except DegenerateGeometryError:
                continue
            if best is None or (inliers, -score) > (best[0], best[1]):
                best = (inliers, -score, fit.x, fit.cov)
        return None if best is None else (best[2], best[3])

    def _score(
        self, x: FloatArray, scoring: list[list[_Pair]], limit: float
    ) -> tuple[int, float]:
        """(# detections explained by some aircraft, sum of their scaled chi-square)."""
        tx = KinematicState(x, _ZERO3)
        k2 = self.config.score_sigma_inflation**2
        inliers, total = 0, 0.0
        for pairs in scoring:
            best = np.inf
            for p in pairs:
                try:
                    y = self._model.residual(p.detection, tx, p.target) / p.sigma
                except DegenerateGeometryError:
                    continue
                best = min(best, float(y @ y) / k2)
            if best <= limit:
                inliers += 1
                total += best
        return inliers, total

    # ------------------------------------------------------------------ #
    # stage 2: associate / refine
    # ------------------------------------------------------------------ #
    def _associate(
        self,
        detections: DetectionSequence,
        tracks: KnownAircraftTracks,
        x: FloatArray,
        cov: FloatArray,
    ) -> list[Assignment]:
        """Best (lowest Mahalanobis) gated known aircraft per detection."""
        prediction = TxPrediction(KinematicState(x, _ZERO3), _cov6(cov))
        best: dict[int, Assignment] = {}
        for c in gate_candidates(
            detections, tracks, self._model, lambda t: prediction, chi2_threshold=self.config.gate_chi2
        ):
            current = best.get(c.detection_index)
            if current is None or c.mahalanobis_sq < current.mahalanobis_sq:
                best[c.detection_index] = Assignment(c.detection_index, c.icao, c.mahalanobis_sq)
        return [best[i] for i in sorted(best)]

    def _pairs(
        self, detections: DetectionSequence, tracks: KnownAircraftTracks, assignments: Sequence[Assignment]
    ) -> list[_Pair]:
        pairs: list[_Pair] = []
        for a in assignments:
            det = detections[a.detection_index]
            target = tracks[a.icao].state_at(det.time_s)
            assert target is not None  # it was gated against this very state
            sigma = np.sqrt(np.diag(self._model.noise_covariance(det)))
            pairs.append(_Pair(a.detection_index, a.icao, det, target, KinematicState.from_detection_receiver(det), sigma))
        return pairs

    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_hint(batch: MeasurementBatch) -> TxLocationHint:
        first = batch.detections.start_time_s
        hint = batch.tx_hints.at_or_before(first) if first is not None else None
        return hint if hint is not None else batch.tx_hints[0]

    @staticmethod
    def _prior_only(hint: TxLocationHint, prior_cov: FloatArray, n_det: int, why: str) -> TxEstimate:
        return TxEstimate(hint.pos_ecef_m, prior_cov, 0.0, (), n_det, np.zeros(0), 0, False, f"returning the hint: {why}")
