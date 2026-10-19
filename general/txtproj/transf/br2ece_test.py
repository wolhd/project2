"""
Round-trip tests for bistatic_to_ecef.

Run with:  python test_bistatic_to_ecef.py     (or: pytest test_bistatic_to_ecef.py)

Each test picks a target, Tx and Rx, computes the bistatic measurement with an
independent forward model, feeds it to the inverse, and checks the original
target position comes back.
"""

import numpy as np

from bistatic_to_ecef import (
    bistatic_to_ecef,
    bistatic_to_ecef_all,
    geodetic_to_ecef,
)

D2R = np.pi / 180.0
TOL_M = 1e-3  # position tolerance [m]


def enu_axes(lat, lon):
    """Independent east/north/up construction for the forward model."""
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return (
        np.array([-so, co, 0.0]),
        np.array([-sl * co, -sl * so, cl]),
        np.array([cl * co, cl * so, sl]),
    )


def forward_measurement(tx, rx, target, tx_lat, tx_lon):
    """Target/Tx/Rx ECEF -> (excess_range, azimuth in [0, 2*pi))."""
    east, north, _ = enu_axes(tx_lat, tx_lon)
    excess = (
        np.linalg.norm(target - tx)
        + np.linalg.norm(target - rx)
        - np.linalg.norm(rx - tx)
    )
    d_rx, d_tg = rx - tx, target - tx
    brg_rx = np.arctan2(d_rx @ east, d_rx @ north)
    brg_tg = np.arctan2(d_tg @ east, d_tg @ north)
    azimuth = (brg_tg - brg_rx) % (2.0 * np.pi)  # clockwise from Tx->Rx LOS
    return excess, azimuth


# --------------------------------------------------------------------------- #
def test_single_example():
    tx_lla = (42.0 * D2R, -71.0 * D2R, 30.0)
    rx_lla = (42.4 * D2R, -71.8 * D2R, 100.0)
    tg_lla = (42.9 * D2R, -70.3 * D2R, 9500.0)

    tx, rx, target = (geodetic_to_ecef(*p) for p in (tx_lla, rx_lla, tg_lla))
    excess, az = forward_measurement(tx, rx, target, tx_lla[0], tx_lla[1])
    print(f"measurement: excess range = {excess:.3f} m, "
          f"azimuth = {np.degrees(az):.4f} deg, altitude = {tg_lla[2]:.1f} m")

    est = bistatic_to_ecef(tx, rx, excess, az, tg_lla[2])
    assert est is not None, "function returned no answer for a valid measurement"
    err = np.linalg.norm(est - target)
    print(f"true  ECEF: {target}\nestim ECEF: {est}\nerror: {err:.2e} m")
    assert err < TOL_M


def test_many_random_geometries():
    rng = np.random.default_rng(1)
    n_ok = n_ambiguous = 0
    for _ in range(300):
        tx_lla = (rng.uniform(30, 55) * D2R, rng.uniform(-100, -60) * D2R, rng.uniform(0, 300))
        # Rx 30-150 km from Tx in a random direction
        b, r = rng.uniform(0, 2 * np.pi), rng.uniform(30e3, 150e3)
        rx_lla = (
            tx_lla[0] + r * np.cos(b) / 6371e3,
            tx_lla[1] + r * np.sin(b) / (6371e3 * np.cos(tx_lla[0])),
            rng.uniform(0, 300),
        )
        # target 20-300 km from Tx in a random direction, 0-15 km altitude
        b, r = rng.uniform(0, 2 * np.pi), rng.uniform(20e3, 300e3)
        tg_lla = (
            tx_lla[0] + r * np.cos(b) / 6371e3,
            tx_lla[1] + r * np.sin(b) / (6371e3 * np.cos(tx_lla[0])),
            rng.uniform(0, 15e3),
        )
        tx, rx, target = (geodetic_to_ecef(*p) for p in (tx_lla, rx_lla, tg_lla))
        excess, az = forward_measurement(tx, rx, target, tx_lla[0], tx_lla[1])

        candidates = bistatic_to_ecef_all(tx, rx, excess, az, tg_lla[2])
        errs = [np.linalg.norm(c - target) for c in candidates]
        assert candidates and min(errs) < TOL_M, "original target not recovered"
        if len(candidates) == 1:
            est = bistatic_to_ecef(tx, rx, excess, az, tg_lla[2])
            assert np.linalg.norm(est - target) < TOL_M
        else:
            n_ambiguous += 1
        n_ok += 1
    print(f"{n_ok} random geometries recovered ({n_ambiguous} had >1 valid solution)")


def test_ambiguous_near_baseline():
    """High target above the Tx-Rx baseline: two valid solutions, both returned."""
    tx_lla = (42.0 * D2R, -71.0 * D2R, 0.0)
    rx_lla = (42.0 * D2R, -69.8 * D2R, 0.0)  # ~99 km due east
    tx, rx = geodetic_to_ecef(*tx_lla), geodetic_to_ecef(*rx_lla)
    tg_lla = (42.0 * D2R, -71.0 * D2R + 0.25 * (rx_lla[1] - tx_lla[1]), 10e3)
    target = geodetic_to_ecef(*tg_lla)
    excess, az = forward_measurement(tx, rx, target, tx_lla[0], tx_lla[1])

    sols = bistatic_to_ecef_all(tx, rx, excess, az, tg_lla[2])
    print(f"near-baseline case returned {len(sols)} candidate(s)")
    assert len(sols) == 2
    assert min(np.linalg.norm(s - target) for s in sols) < TOL_M


def test_unrealistic_measurements_return_none():
    tx = geodetic_to_ecef(42.0 * D2R, -71.0 * D2R, 30.0)
    rx = geodetic_to_ecef(42.4 * D2R, -71.8 * D2R, 100.0)

    # negative excess range is impossible (triangle inequality)
    assert bistatic_to_ecef(tx, rx, -100.0, 1.0, 5000.0) is None
    # zero excess range requires the target on the baseline, impossible at 10 km altitude
    assert bistatic_to_ecef(tx, rx, 0.0, 0.0, 10e3) is None
    assert bistatic_to_ecef(tx, rx, 0.0, 1.5, 10e3) is None
    # excess range larger than anything reachable on the Earth at that altitude
    assert bistatic_to_ecef(tx, rx, 5e7, 1.0, 10e3) is None
    # non-finite inputs
    assert bistatic_to_ecef(tx, rx, np.nan, 1.0, 10e3) is None
    assert bistatic_to_ecef(tx, rx, 50e3, np.inf, 10e3) is None
    # Tx == Rx: azimuth reference undefined
    assert bistatic_to_ecef(tx, tx, 50e3, 1.0, 10e3) is None
    print("unrealistic measurements correctly returned None")


if __name__ == "__main__":
    test_single_example()
    test_many_random_geometries()
    test_ambiguous_near_baseline()
    test_unrealistic_measurements_return_none()
    print("\nAll tests passed.")
