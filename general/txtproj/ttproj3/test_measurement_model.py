from __future__ import annotations

import math

import numpy as np
import pytest

from txloc import (
    BistaticMeasurementModel,
    DegenerateGeometryError,
    DetectionSequence,
    KinematicState,
    KnownAircraftSequence,
    KnownAircraftState,
    KnownAircraftTrack,
    KnownAircraftTracks,
    MeasurementComponent as C,
    TxPrediction,
    gate_candidates,
)
from txloc.geodesy import ecef_to_geodetic, ellipsoid_frame, enu_to_ecef_rotation, geodetic_to_ecef

LAT, LON = 40.0, -100.0
ORIGIN = geodetic_to_ecef(LAT, LON, 0.0)
ROT = enu_to_ecef_rotation(LAT, LON)


def enu(e: float, n: float, u: float) -> np.ndarray:
    """ECEF point at the given ENU offset from ORIGIN."""
    return ORIGIN + ROT @ np.array([e, n, u], dtype=float)


def vel(e: float, n: float, u: float) -> np.ndarray:
    return ROT @ np.array([e, n, u], dtype=float)


TX = KinematicState(enu(-80e3, 20e3, 300.0), vel(10.0, 5.0, 0.0))
TGT = KinematicState(enu(40e3, 30e3, 9e3), vel(150.0, -40.0, 5.0))
RX = KinematicState(ORIGIN, vel(20.0, 0.0, 0.0))
MODEL = BistaticMeasurementModel()


# ---- model --------------------------------------------------------------- #
def test_jacobian_matches_finite_differences() -> None:
    h = MODEL.jacobian(TX, TGT, RX)
    assert h.shape == (3, 6)
    x0 = TX.to_vector()
    fd = np.zeros((3, 6))
    for k in range(6):
        step = np.zeros(6)
        step[k] = 1.0
        hi = MODEL.predict(KinematicState.from_vector(x0 + step), TGT, RX)
        lo = MODEL.predict(KinematicState.from_vector(x0 - step), TGT, RX)
        fd[:, k] = [
            (hi.range_m - lo.range_m) / 2.0,
            (hi.range_rate_mps - lo.range_rate_mps) / 2.0,
            (hi.azimuth_rad - lo.azimuth_rad) / 2.0,
        ]
    np.testing.assert_allclose(h[0], fd[0], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(h[1], fd[1], rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(h[2], fd[2], rtol=1e-4, atol=1e-11)
    assert np.all(h[2, 3:] == 0.0)  # azimuth does not depend on tx velocity
    assert np.any(h[2, :3] != 0.0)  # but it does depend on tx position


def test_range_rate_is_time_derivative_of_excess_range() -> None:
    dt = 0.01

    def rng(t: float) -> float:
        return MODEL.predict(TX.propagate(t), TGT.propagate(t), RX.propagate(t)).range_m

    numeric = (rng(dt) - rng(-dt)) / (2 * dt)
    assert MODEL.predict(TX, TGT, RX).range_rate_mps == pytest.approx(numeric, rel=1e-6)


def test_excess_range_definition() -> None:
    pred = MODEL.predict(TX, TGT, RX)
    expected = (
        np.linalg.norm(TGT.pos_ecef_m - TX.pos_ecef_m)
        + np.linalg.norm(TGT.pos_ecef_m - RX.pos_ecef_m)
        - np.linalg.norm(TX.pos_ecef_m - RX.pos_ecef_m)
    )
    assert pred.range_m == pytest.approx(expected)
    on_baseline = KinematicState((TX.pos_ecef_m + RX.pos_ecef_m) / 2, [0, 0, 0])
    assert MODEL.predict(TX, on_baseline, RX).range_m == pytest.approx(0.0, abs=1e-6)


def _azimuth(tx_enu: tuple[float, float, float], tgt_enu: tuple[float, float, float],
             rx_enu: tuple[float, float, float]) -> float:
    zero = [0.0, 0.0, 0.0]
    return MODEL.predict(
        KinematicState(enu(*tx_enu), zero), KinematicState(enu(*tgt_enu), zero), KinematicState(enu(*rx_enu), zero)
    ).azimuth_rad


def test_azimuth_is_clockwise_from_tx_to_rx_line_of_sight() -> None:
    tx = (0.0, 0.0, 0.0)
    rx_north = (0.0, 10e3, 0.0)
    # LOS (tx->rx) points north
    assert _azimuth(tx, (0.0, 5e3, 0.0), rx_north) == pytest.approx(0.0, abs=1e-9)  # along LOS
    assert _azimuth(tx, (5e3, 0.0, 0.0), rx_north) == pytest.approx(math.pi / 2, abs=1e-3)  # east = clockwise
    assert _azimuth(tx, (-5e3, 0.0, 0.0), rx_north) == pytest.approx(-math.pi / 2, abs=1e-3)
    assert abs(_azimuth(tx, (0.0, -5e3, 0.0), rx_north)) == pytest.approx(math.pi, abs=1e-3)
    # LOS points east, target south: south is 90 deg clockwise from east
    assert _azimuth(tx, (0.0, -5e3, 0.0), (10e3, 0.0, 0.0)) == pytest.approx(math.pi / 2, abs=1e-3)
    # height does not matter, only the horizontal direction
    assert _azimuth(tx, (5e3, 0.0, 8e3), rx_north) == pytest.approx(math.pi / 2, abs=1e-3)


def test_ellipsoid_frame_up_jacobian_matches_finite_differences() -> None:
    p = geodetic_to_ecef(52.0, 13.0, 9_000.0)
    jac = ellipsoid_frame(p).up_jacobian
    fd = np.zeros((3, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = 5.0
        fd[:, k] = (ellipsoid_frame(p + step).up - ellipsoid_frame(p - step).up) / 10.0
    np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(jac, jac.T)


@pytest.mark.parametrize("lat", [-89.9, -45.0, 0.0, 33.3, 89.9])
def test_ecef_geodetic_round_trip(lat: float) -> None:
    for alt in (-100.0, 0.0, 11_000.0):
        out = ecef_to_geodetic(geodetic_to_ecef(lat, 77.0, alt))
        assert out == pytest.approx((lat, 77.0, alt), abs=1e-6)


def test_simulate_residual_and_gate() -> None:
    det = MODEL.simulate_detection(1.0, TX, TGT, RX, range_std_m=50, azimuth_std_rad=0.02, range_rate_std_mps=2)
    np.testing.assert_allclose(MODEL.residual(det, TX, TGT), 0.0, atol=1e-6)

    cov = np.diag([500.0**2] * 3 + [20.0**2] * 3)
    assert MODEL.gate(det, TX, TGT, cov)
    assert MODEL.mahalanobis_sq(det, TX, TGT, cov) < 1e-9
    far = KinematicState(TX.pos_ecef_m + vel(30e3, 0, 0), TX.vel_ecef_mps)
    assert not MODEL.gate(det, far, TGT, cov)

    rng = np.random.default_rng(1)
    z = np.array(
        [
            MODEL.residual(
                MODEL.simulate_detection(0, TX, TGT, RX, range_std_m=50, azimuth_std_rad=0.02,
                                         range_rate_std_mps=2, rng=rng), TX, TGT)
            for _ in range(4000)
        ]
    )
    np.testing.assert_allclose(z.std(axis=0), [50, 2, 0.02], rtol=0.06)


def test_validation_and_degenerate_geometry() -> None:
    with pytest.raises(TypeError):
        MODEL.predict(TX.to_vector(), TGT, RX)  # type: ignore[arg-type]
    with pytest.raises(DegenerateGeometryError):
        MODEL.predict(TX, KinematicState(TX.pos_ecef_m, [0, 0, 0]), RX)
    above = KinematicState(TX.pos_ecef_m + ellipsoid_frame(TX.pos_ecef_m).up * 5e3, [0, 0, 0])
    with pytest.raises(DegenerateGeometryError):  # target straight above tx: azimuth undefined
        MODEL.predict(TX, above, RX)
    with pytest.raises(ValueError):
        BistaticMeasurementModel(components=(C.RANGE, C.RANGE))
    with pytest.raises(TypeError):
        BistaticMeasurementModel(components=("range",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        KinematicState.from_vector([1, 2, 3])


def test_component_selection_orders_rows() -> None:
    m = BistaticMeasurementModel(components=(C.AZIMUTH, C.RANGE))
    full = MODEL.jacobian(TX, TGT, RX)
    np.testing.assert_allclose(m.jacobian(TX, TGT, RX), full[[2, 0]])


# ---- tracks -------------------------------------------------------------- #
def _track_states(icao: str = "AAA111") -> KnownAircraftSequence:
    p0, v0, a = np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])

    def at(t: float) -> KnownAircraftState:
        return KnownAircraftState(t, icao, p0 + v0 * t + 0.5 * a * t * t, v0 + a * t)

    return KnownAircraftSequence([at(0.0), at(4.0), at(30.0)])


def test_hermite_reproduces_constant_acceleration_exactly() -> None:
    track = KnownAircraftTrack(_track_states(), max_gap_s=10.0)
    s = track.state_at(1.7)
    assert s is not None
    np.testing.assert_allclose(s.pos_ecef_m, [170.0, 0.5 * 2 * 1.7**2, 0.0], atol=1e-9)
    np.testing.assert_allclose(s.vel_ecef_mps, [100.0, 2 * 1.7, 0.0], atol=1e-9)


def test_track_gap_and_extrapolation_limits() -> None:
    track = KnownAircraftTrack(_track_states(), max_gap_s=10.0, max_extrapolation_s=5.0)
    assert track.state_at(10.0) is None  # inside the 4..30 s gap
    assert track.state_at(33.0) is not None and track.state_at(36.0) is None
    assert track.state_at(-4.0) is not None and track.state_at(-6.0) is None


def test_linear_fallback_without_velocity() -> None:
    seq = KnownAircraftSequence(
        [KnownAircraftState(0, "B", [0, 0, 0]), KnownAircraftState(2, "B", [10, 0, 0])]
    )
    s = KnownAircraftTrack(seq).state_at(1.0)
    assert s is not None
    np.testing.assert_allclose(s.pos_ecef_m, [5, 0, 0])
    np.testing.assert_allclose(s.vel_ecef_mps, [5, 0, 0])


def test_track_requires_single_aircraft() -> None:
    both = KnownAircraftSequence([KnownAircraftState(0, "A", [0, 0, 0]), KnownAircraftState(1, "B", [0, 0, 0])])
    with pytest.raises(ValueError):
        KnownAircraftTrack(both)
    assert KnownAircraftTracks.from_sequence(both).icaos == ("A", "B")


# ---- association --------------------------------------------------------- #
def _tracks() -> KnownAircraftTracks:
    def ac(icao: str, e: list[float]) -> list[KnownAircraftState]:
        p = enu(*e)
        v = vel(100.0, 0.0, 0.0)
        return [KnownAircraftState(t, icao, p + v * t, v) for t in (0.0, 10.0)]

    return KnownAircraftTracks.from_sequence(
        KnownAircraftSequence(ac("NEAR01", [40e3, 30e3, 9e3]) + ac("OTHER1", [-20e3, 60e3, 8e3]))
    )


def test_gate_candidates_pick_matching_aircraft() -> None:
    tracks = _tracks()
    target = tracks["NEAR01"].state_at(5.0)
    assert target is not None
    det = MODEL.simulate_detection(5.0, TX, target, RX, range_std_m=50, azimuth_std_rad=0.02, range_rate_std_mps=2)
    cov = np.diag([500.0**2] * 3 + [20.0**2] * 3)
    seq = DetectionSequence([det])

    good = gate_candidates(seq, tracks, MODEL, lambda t: TxPrediction(TX, cov))
    assert [c.icao for c in good] == ["NEAR01"]
    assert good[0].mahalanobis_sq < 1e-6 and good[0].detection_index == 0

    off = KinematicState(TX.pos_ecef_m + vel(30e3, 0, 0), TX.vel_ecef_mps)
    assert gate_candidates(seq, tracks, MODEL, lambda t: TxPrediction(off, cov)) == []
    with pytest.raises(TypeError):
        gate_candidates(seq, tracks, "model", lambda t: TxPrediction(TX, cov))  # type: ignore[arg-type]
