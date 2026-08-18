"""Exercises transform -> interpolation -> association -> tracking with the
PassThrough stubs, without touching zmq or the filesystem.
"""

from trackerapp.association.base import PassThroughAssociator
from trackerapp.interpolation.interpolator import Interpolator
from trackerapp.models.types import Detection, GeodeticPosition
from trackerapp.tracking.base import PassThroughTracker
from trackerapp.transforms.geodetic_ecef import geodetic_to_ecef


def test_stages_chain_together():
    interpolator = Interpolator(step_s=1.0, max_gap_s=10.0, method="linear")
    associator = PassThroughAssociator()
    tracker = PassThroughTracker()

    reports = [
        GeodeticPosition("t1", 42.36, -71.06, 10.0, 0.0),
        GeodeticPosition("t1", 42.361, -71.059, 10.0, 2.0),
    ]

    results = []
    for report in reports:
        ecef = geodetic_to_ecef(report)
        for interpolated in interpolator.push(ecef):
            detection = Detection(position=interpolated)
            associated = associator.associate(detection)
            results.append(tracker.update(associated))

    # 2 originals + 1 interpolated midpoint at t=1.0
    assert len(results) == 3
    assert [round(r.position.timestamp, 6) for r in results] == [0.0, 1.0, 2.0]
    assert all(r.track_uid is not None for r in results)
