"""Input handling for bistatic transmitter localization (data models, CSV I/O, playback, buffering)."""

from .association import AssociationCandidate, TxPrediction, TxPredictor, gate_candidates
from .buffer import MeasurementBuffer
from .csv_io import CsvFormatError, read_detections_csv, read_known_aircraft_csv, read_tx_hint_csv
from .estimator import Assignment, EstimatorConfig, StaticTxEstimator, TxEstimate
from .kinematics import KinematicState, KnownAircraftTrack, KnownAircraftTracks
from .measurement_model import (
    CHI2_99,
    BistaticMeasurementModel,
    DegenerateGeometryError,
    MeasurementComponent,
    PredictedMeasurement,
)
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
)
from .playback import MeasurementPlayback, MeasurementSink

__all__ = [
    "Assignment",
    "EstimatorConfig",
    "StaticTxEstimator",
    "TxEstimate",
    "AssociationCandidate",
    "BistaticMeasurementModel",
    "CHI2_99",
    "DegenerateGeometryError",
    "KinematicState",
    "KnownAircraftTrack",
    "KnownAircraftTracks",
    "MeasurementComponent",
    "PredictedMeasurement",
    "TxPrediction",
    "TxPredictor",
    "gate_candidates",
    "BistaticDetection",
    "CsvFormatError",
    "DetectionSequence",
    "KnownAircraftSequence",
    "KnownAircraftState",
    "Measurement",
    "MeasurementBatch",
    "MeasurementBuffer",
    "MeasurementKind",
    "MeasurementPlayback",
    "MeasurementSink",
    "TxHintSequence",
    "TxLocationHint",
    "read_detections_csv",
    "read_known_aircraft_csv",
    "read_tx_hint_csv",
]
