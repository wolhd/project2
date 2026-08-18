"""Ingest stage interface.

Any source of position reports (zmq subscriber, json file, future
replacements like a rosbag reader or a REST poller) implements this and
yields ``GeodeticPosition`` objects. The pipeline only depends on this
interface, never on a concrete reader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from trackerapp.models.types import GeodeticPosition


class SourceReader(ABC):
    """Yields decoded geodetic position reports."""

    @abstractmethod
    def __iter__(self) -> Iterator[GeodeticPosition]:
        """Iterate over available position reports, blocking as needed."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (sockets, file handles). Optional to override."""
        return None

    def __enter__(self) -> "SourceReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
