"""Association stage interface.

The real implementation will come from a future library (e.g. a
gating/nearest-neighbor or JPDA associator). Wire it in by writing an
adapter class that implements ``Associator`` and updating
``pipeline.py``'s construction call - no other code should need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trackerapp.models.types import AssociatedDetection, Detection


class Associator(ABC):
    """Associates a new detection with an existing track, or flags it as new."""

    @abstractmethod
    def associate(self, detection: Detection) -> AssociatedDetection:
        """Return the detection tagged with a track_uid (or None if unmatched)."""
        raise NotImplementedError


class PassThroughAssociator(Associator):
    """Placeholder associator used until the real library is integrated.

    Assigns a new, never-reused track_uid to every detection - i.e. no
    actual association occurs. Useful for exercising the rest of the
    pipeline end-to-end before the association library is available.
    """

    def __init__(self) -> None:
        self._next_uid = 1

    def associate(self, detection: Detection) -> AssociatedDetection:
        uid = self._next_uid
        self._next_uid += 1
        return AssociatedDetection(detection=detection, track_uid=uid)
