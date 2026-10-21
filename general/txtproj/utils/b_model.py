import numpy as np
import pymap3d as pm
from dataclasses import dataclass


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bistatic_model(x):
    """
    x = [tgt_pos(3), tgt_vel(3), tx_pos(3), tx_vel(3), rx_pos(3), rx_vel(3)]  (ECEF, m and m/s)
    returns [excess_range (m), excess_range_rate (m/s), azimuth (rad, wrapped to [-pi, pi))]
    """
    pt, vt = x[0:3], x[3:6]
    ptx, vtx = x[6:9], x[9:12]
    prx, vrx = x[12:15], x[15:18]

    d_tx = pt - ptx          # tx -> target
    d_rx = pt - prx          # rx -> target
    base = prx - ptx         # tx -> rx
    R_tx, R_rx, L = map(np.linalg.norm, (d_tx, d_rx, base))

    excess_range = R_tx + R_rx - L
    excess_rate = (d_tx @ (vt - vtx)) / R_tx \
                + (d_rx @ (vt - vrx)) / R_rx \
                - (base @ (vrx - vtx)) / L

    # local ENU frame at the transmitter (WGS-84 via pymap3d)
    lat0, lon0, h0 = pm.ecef2geodetic(*ptx, deg=False)
    e_t, n_t, _ = pm.ecef2enu(*pt,  lat0, lon0, h0, deg=False)
    e_r, n_r, _ = pm.ecef2enu(*prx, lat0, lon0, h0, deg=False)

    # bearing = atan2(E, N) is clockwise from north; subtract the tx->rx bearing
    az = _wrap_pi(np.arctan2(e_t, n_t) - np.arctan2(e_r, n_r))
    return np.array([excess_range, excess_rate, az])


def _jacobian(x, h_pos=1.0, h_vel=0.01):
    """Central-difference Jacobian (3 x 18); azimuth differences are wrap-aware."""
    J = np.zeros((3, x.size))
    h0 = bistatic_model(x)
    for i in range(x.size):
        step = h_pos if (i // 3) % 2 == 0 else h_vel   # blocks: pos, vel, pos, vel, ...
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        hp, hm = bistatic_model(xp), bistatic_model(xm)
        hp[2] = h0[2] + _wrap_pi(hp[2] - h0[2])
        hm[2] = h0[2] + _wrap_pi(hm[2] - h0[2])
        J[:, i] = (hp - hm) / (2 * step)
    return J


@dataclass
class BistaticMeasurement:
    excess_range: float       # m
    range_rate: float         # m/s
    azimuth: float            # rad in [0, 2pi), clockwise from tx->rx LOS
    var_range: float
    var_range_rate: float
    var_azimuth: float
    cov: np.ndarray           # 3x3 measurement covariance
    jacobian: np.ndarray      # 3x18, columns ordered [target, tx, rx] x [pos, vel]


def bistatic_measurement(target, tx, rx, P_target=None, P_tx=None, P_rx=None, P_full=None):
    """
    target, tx, rx : length-6 [x, y, z, vx, vy, vz] in ECEF
    P_*            : 6x6 covariances (assumed independent), or
    P_full         : 18x18 joint covariance ordered [target, tx, rx] (overrides P_*)
    """
    x = np.concatenate([np.asarray(a, float) for a in (target, tx, rx)])
    z = bistatic_model(x)

    if P_full is None:
        P_full = np.zeros((18, 18))
        for k, P in enumerate((P_target, P_tx, P_rx)):
            if P is not None:
                P_full[6*k:6*k+6, 6*k:6*k+6] = P

    J = _jacobian(x)
    R = J @ P_full @ J.T
    return BistaticMeasurement(z[0], z[1], z[2] % (2 * np.pi),
                               R[0, 0], R[1, 1], R[2, 2], R, J)


if __name__ == "__main__":
    tgt = [1_100_000, -4_300_000, 4_400_000, 200, -50, 10]
    tx  = [1_080_000, -4_320_000, 4_400_000, 5, 3, 0]
    rx  = [1_130_000, -4_290_000, 4_390_000, 100, 0, 0]

    P_t  = np.diag([50**2]*3 + [5**2]*3)
    P_tx = np.diag([100**2]*3 + [1**2]*3)
    P_rx = np.diag([5**2]*3 + [0.1**2]*3)

    m = bistatic_measurement(tgt, tx, rx, P_t, P_tx, P_rx)
    print(f"excess range : {m.excess_range:.1f} m    (sigma {np.sqrt(m.var_range):.2f})")
    print(f"range rate   : {m.range_rate:.2f} m/s  (sigma {np.sqrt(m.var_range_rate):.3f})")
    print(f"azimuth      : {np.degrees(m.azimuth):.3f} deg (sigma {np.degrees(np.sqrt(m.var_azimuth)):.4f})")
