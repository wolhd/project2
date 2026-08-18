"""Output stage interface.

Publishers receive a ``TrackedState`` (ECEF) from the tracking stage,
convert it back to geodetic for the outgoing message, and publish it via
zmq or write it to a json file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trackerapp.models.types import TrackedState
from trackerapp.transforms.geodetic_ecef import ecef_to_geodetic


class ResultPublisher(ABC):
    """Publishes a tracked state to some sink (zmq, file, ...)."""

    @abstractmethod
    def publish(self, state: TrackedState) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources. Optional to override."""
        return None

    def __enter__(self) -> "ResultPublisher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def to_geodetic_dict(state: TrackedState) -> dict:
    """Shared helper: convert a TrackedState (ECEF) to the outgoing field set."""
    geo = ecef_to_geodetic(state.position)
    return {
        "track_id": geo.track_id,
        "track_uid": state.track_uid,
        "latitude_deg": geo.latitude_deg,
        "longitude_deg": geo.longitude_deg,
        "altitude_m": geo.altitude_m,
        "timestamp": geo.timestamp,
        "confidence": state.confidence,
    }
