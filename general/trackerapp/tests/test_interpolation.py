from trackerapp.interpolation.interpolator import Interpolator
from trackerapp.models.types import EcefPosition


def test_interpolates_between_two_samples():
    interp = Interpolator(step_s=1.0, max_gap_s=10.0, method="linear")

    p0 = EcefPosition(track_id="a", x_m=0.0, y_m=0.0, z_m=0.0, timestamp=0.0)
    p1 = EcefPosition(track_id="a", x_m=10.0, y_m=0.0, z_m=0.0, timestamp=4.0)

    out0 = list(interp.push(p0))
    assert out0 == [p0]

    out1 = list(interp.push(p1))
    # expect points at t=1,2,3 then the original t=4 sample
    assert [round(p.timestamp, 6) for p in out1] == [1.0, 2.0, 3.0, 4.0]
    assert out1[0].x_m == 2.5
    assert out1[-1] is p1


def test_gap_too_large_is_not_bridged():
    interp = Interpolator(step_s=1.0, max_gap_s=2.0, method="linear")
    p0 = EcefPosition(track_id="a", x_m=0.0, y_m=0.0, z_m=0.0, timestamp=0.0)
    p1 = EcefPosition(track_id="a", x_m=10.0, y_m=0.0, z_m=0.0, timestamp=10.0)

    list(interp.push(p0))
    out1 = list(interp.push(p1))
    assert out1 == [p1]


def test_separate_tracks_do_not_interfere():
    interp = Interpolator(step_s=1.0, max_gap_s=10.0, method="linear")
    a0 = EcefPosition(track_id="a", x_m=0.0, y_m=0.0, z_m=0.0, timestamp=0.0)
    b0 = EcefPosition(track_id="b", x_m=100.0, y_m=0.0, z_m=0.0, timestamp=0.0)

    list(interp.push(a0))
    out_b = list(interp.push(b0))
    # first sample of a new track_id is always passed through untouched
    assert out_b == [b0]
