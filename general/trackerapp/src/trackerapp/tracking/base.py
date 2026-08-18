"""State tracking stage interface.

The real implementation will come from a future library (e.g. a Kalman /
particle filter track manager). Wire it in by writing an adapter class that
implements ``StateTracker`` and updating ``pipeline.py``'s construction call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trackerapp.models.types import AssociatedDetection, TrackedState


class StateTracker(ABC):
    """Consumes associated detections and produces filtered track states."""

    @abstractmethod
    def update(self, associated: AssociatedDetection) -> TrackedState:
        """Update (or initialize) the track's state estimate with a new detection."""
        raise NotImplementedError


class PassThroughTracker(StateTracker):
    """Placeholder tracker used until the real library is integrated.

    Emits the raw associated position as the "tracked" state with no
    filtering. Useful for exercising the rest of the pipeline end-to-end
    before the tracking library is available.
    """

    def update(self, associated: AssociatedDetection) -> TrackedState:
        track_uid = associated.track_uid if associated.track_uid is not None else 0
        return TrackedState(
            track_uid=track_uid,
            position=associated.detection.position,
            velocity_mps=None,
            confidence=1.0,
        )
