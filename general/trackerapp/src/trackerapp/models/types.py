"""Internal data types shared across all pipeline stages.

Ingest stages produce ``GeodeticPosition``. Everything from the transform
stage onward (interpolation, association, tracking) operates on
``EcefPosition``. The output wrapper is responsible for converting the final
``TrackedState`` back to geodetic before publishing, if the sink needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GeodeticPosition:
    """A raw lat/lon/alt/time report, as decoded from protobuf or json."""

    track_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    timestamp: float  # unix epoch seconds


@dataclass(slots=True)
class EcefPosition:
    """A position in Earth-Centered, Earth-Fixed cartesian coordinates.

    This is the internal representation used by interpolation, association,
    and tracking. Carries the originating track_id/timestamp through so
    downstream stages don't need a separate lookup.
    """

    track_id: str
    x_m: float
    y_m: float
    z_m: float
    timestamp: float


@dataclass(slots=True)
class Detection:
    """A single association-ready measurement (post-interpolation)."""

    position: EcefPosition
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class AssociatedDetection:
    """Output of the association stage: a detection tied to a track hypothesis."""

    detection: Detection
    track_uid: int | None  # None => unassociated / new track candidate


@dataclass(slots=True)
class TrackedState:
    """Output of the state-tracking stage: a filtered/estimated state in ECEF."""

    track_uid: int
    position: EcefPosition
    velocity_mps: tuple[float, float, float] | None = None
    confidence: float = 1.0
