"""Writes TrackedState results to a json file as newline-delimited json."""

from __future__ import annotations

import json
from pathlib import Path

from trackerapp.models.types import TrackedState
from trackerapp.output.base import ResultPublisher, to_geodetic_dict


class JsonFileWriter(ResultPublisher):
    """Appends one json object per line to the configured output path."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a")

    def publish(self, state: TrackedState) -> None:
        record = to_geodetic_dict(state)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
