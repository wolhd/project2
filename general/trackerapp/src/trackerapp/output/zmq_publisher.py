"""Publishes TrackedPosition protobuf messages on a zmq PUB socket."""

from __future__ import annotations

import logging

import zmq

from trackerapp.models.types import TrackedState
from trackerapp.output.base import ResultPublisher, to_geodetic_dict

try:
    from trackerapp.proto_gen import position_pb2
except ImportError:  # pragma: no cover - only hit before codegen has run
    position_pb2 = None

logger = logging.getLogger(__name__)


class ZmqResultPublisher(ResultPublisher):
    """Binds a zmq PUB socket and publishes TrackedPosition protobuf messages."""

    def __init__(self, endpoint: str, topic: str = "tracked") -> None:
        if position_pb2 is None:
            raise RuntimeError(
                "position_pb2 not found - run scripts/generate_proto.sh first"
            )
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(endpoint)
        self._topic = topic.encode()
        logger.info("publishing on %s (topic=%r)", endpoint, topic)

    def publish(self, state: TrackedState) -> None:
        fields = to_geodetic_dict(state)
        msg = position_pb2.TrackedPosition(
            track_id=fields["track_id"],
            latitude_deg=fields["latitude_deg"],
            longitude_deg=fields["longitude_deg"],
            altitude_m=fields["altitude_m"],
            timestamp=fields["timestamp"],
            track_uid=fields["track_uid"],
            confidence=fields["confidence"],
        )
        self._socket.send_multipart([self._topic, msg.SerializeToString()])

    def close(self) -> None:
        self._socket.close()
