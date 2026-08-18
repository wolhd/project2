"""Wires ingest -> transform -> interpolation -> association -> tracking ->
output into a single run loop.

This module owns stage *construction* (picking zmq vs json_file per config)
and the *run loop*. It intentionally knows nothing about how any stage
does its job internally - swap an implementation by changing a constructor
call here.
"""

from __future__ import annotations

import logging

from trackerapp.association.base import Associator, PassThroughAssociator
from trackerapp.config import AppConfig
from trackerapp.ingest.base import SourceReader
from trackerapp.ingest.json_file_reader import JsonFileReader
from trackerapp.ingest.zmq_reader import ZmqPositionReader
from trackerapp.interpolation.interpolator import Interpolator
from trackerapp.models.types import Detection
from trackerapp.output.base import ResultPublisher
from trackerapp.output.json_file_writer import JsonFileWriter
from trackerapp.output.zmq_publisher import ZmqResultPublisher
from trackerapp.tracking.base import PassThroughTracker, StateTracker
from trackerapp.transforms.geodetic_ecef import geodetic_to_ecef

logger = logging.getLogger(__name__)


def build_source_reader(config: AppConfig) -> SourceReader:
    if config.ingest.mode == "zmq":
        return ZmqPositionReader(config.ingest.zmq_endpoint, config.ingest.zmq_topic)
    if config.ingest.mode == "json_file":
        return JsonFileReader(config.ingest.json_path, config.ingest.json_poll_interval_s)
    raise ValueError(f"unknown ingest mode: {config.ingest.mode}")


def build_result_publisher(config: AppConfig) -> ResultPublisher:
    if config.output.mode == "zmq":
        return ZmqResultPublisher(config.output.zmq_endpoint, config.output.zmq_topic)
    if config.output.mode == "json_file":
        return JsonFileWriter(config.output.json_path)
    raise ValueError(f"unknown output mode: {config.output.mode}")


def build_associator(config: AppConfig) -> Associator:
    # TODO: replace with the real association library adapter once available.
    return PassThroughAssociator()


def build_tracker(config: AppConfig) -> StateTracker:
    # TODO: replace with the real state-tracking library adapter once available.
    return PassThroughTracker()


class Pipeline:
    """Owns one instance of each stage and runs the ingest -> output loop."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._reader = build_source_reader(config)
        self._publisher = build_result_publisher(config)
        self._interpolator = Interpolator(
            step_s=config.interpolation.step_s,
            max_gap_s=config.interpolation.max_gap_s,
            method=config.interpolation.method,
        )
        self._associator = build_associator(config)
        self._tracker = build_tracker(config)

    def run(self) -> None:
        with self._reader, self._publisher:
            for geodetic in self._reader:
                self._process_one(geodetic)

    def _process_one(self, geodetic) -> None:
        ecef = geodetic_to_ecef(geodetic)
        for interpolated in self._interpolator.push(ecef):
            detection = Detection(position=interpolated)
            associated = self._associator.associate(detection)
            tracked_state = self._tracker.update(associated)
            self._publisher.publish(tracked_state)
