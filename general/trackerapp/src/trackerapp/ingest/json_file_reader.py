"""Reads Position reports from a json file.

Supports two formats, auto-detected from the parsed JSON:
  - a top-level list of objects: [{"track_id": ..., "latitude_deg": ...}, ...]
  - newline-delimited json (one object per line)

Field names match proto/position.proto so the same file can be produced by
dumping decoded protobuf messages to json for offline replay.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

from trackerapp.ingest.base import SourceReader
from trackerapp.models.types import GeodeticPosition

logger = logging.getLogger(__name__)


class JsonFileReader(SourceReader):
    """Reads position reports from a json or ndjson file.

    If ``poll_interval_s`` is 0, reads the file once and stops (offline
    replay). If > 0, treats the file as line-delimited json being appended
    to live and polls for new lines (tail -f style).
    """

    def __init__(self, path: str | Path, poll_interval_s: float = 0.0) -> None:
        self._path = Path(path)
        self._poll_interval_s = poll_interval_s

    def __iter__(self) -> Iterator[GeodeticPosition]:
        if self._poll_interval_s > 0:
            yield from self._tail()
        else:
            yield from self._read_once()

    def _read_once(self) -> Iterator[GeodeticPosition]:
        text = self._path.read_text().strip()
        if not text:
            return
        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for record in records:
            yield _record_to_position(record)

    def _tail(self) -> Iterator[GeodeticPosition]:
        with self._path.open("r") as fh:
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(self._poll_interval_s)
                    continue
                line = line.strip()
                if not line:
                    continue
                yield _record_to_position(json.loads(line))


def _record_to_position(record: dict) -> GeodeticPosition:
    return GeodeticPosition(
        track_id=str(record.get("track_id", "")),
        latitude_deg=float(record["latitude_deg"]),
        longitude_deg=float(record["longitude_deg"]),
        altitude_m=float(record.get("altitude_m", 0.0)),
        timestamp=float(record["timestamp"]),
    )
