"""Candidate association of detections with known aircraft.

Every component of a detection (range, range rate and, since azimuth is an
angle at the transmitter, azimuth too) depends on the transmitter state, so
gating needs a predicted transmitter state and covariance at each detection
time.  The estimator (or a simple hint-based predictor) supplies those through
a ``TxPredictor``.

A passing gate means "possible", not "certain": with many unknown aircraft and
clutter, several detections may fit one aircraft and vice versa, so all
candidates are returned for the estimator to resolve (nearest neighbour, PDA,
multi-hypothesis ...).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from .kinematics import KinematicState, KnownAircraftTracks
from .measurement_model import CHI2_99, BistaticMeasurementModel, DegenerateGeometryError
from .models import DetectionSequence
from .validation import FloatArray, as_positive_float, as_square_matrix


class TxPrediction(NamedTuple):
    """Predicted transmitter state and its 6x6 covariance at one time."""

    state: KinematicState
    cov: FloatArray


TxPredictor = Callable[[float], TxPrediction]


@dataclass(frozen=True, slots=True, eq=False)
class AssociationCandidate:
    detection_index: int  # index into the DetectionSequence
    icao: str
    mahalanobis_sq: float
    residual: FloatArray  # measured - predicted, in the model's component order


def gate_candidates(
    detections: DetectionSequence,
    tracks: KnownAircraftTracks,
    model: BistaticMeasurementModel,
    tx_predictor: TxPredictor,
    *,
    chi2_threshold: float | None = None,
) -> list[AssociationCandidate]:
    """All (detection, known aircraft) pairs passing the model's chi-square gate.

    Aircraft whose track cannot support the detection time, and geometries that
    are degenerate for the model, are skipped.  ``chi2_threshold`` defaults to
    the 99 % quantile for the model's dimension.
    """
    if not isinstance(detections, DetectionSequence):
        raise TypeError(f"detections must be DetectionSequence, got {type(detections).__name__}")
    if not isinstance(tracks, KnownAircraftTracks):
        raise TypeError(f"tracks must be KnownAircraftTracks, got {type(tracks).__name__}")
    if not isinstance(model, BistaticMeasurementModel):
        raise TypeError(f"model must be BistaticMeasurementModel, got {type(model).__name__}")
    limit = CHI2_99[model.dim] if chi2_threshold is None else as_positive_float(chi2_threshold, "chi2_threshold")

    out: list[AssociationCandidate] = []
    for idx, det in enumerate(detections):
        prediction = tx_predictor(det.time_s)
        tx_cov = as_square_matrix(prediction.cov, model.TX_STATE_DIM, "tx prediction covariance")
        for icao, target in tracks.states_at(det.time_s).items():
            try:
                d2 = model.mahalanobis_sq(det, prediction.state, target, tx_cov)
                residual = model.residual(det, prediction.state, target)
            except DegenerateGeometryError:
                continue
            if d2 <= limit:
                out.append(AssociationCandidate(idx, icao, d2, residual))
    return out
