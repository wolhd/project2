"""
Convert a bistatic radar measurement into an ECEF target position.

Measurement definition
----------------------
excess_range : (|T - Tx| + |T - Rx|) - |Rx - Tx|   [m], must be >= 0
azimuth      : measured at the transmitter, clockwise (seen from above) from the
               horizontal direction of the Tx->Rx line of sight to the horizontal
               direction of the Tx->target line of sight            [rad]
altitude     : target geodetic height above the WGS-84 ellipsoid     [m]

Geometry
--------
There is no elevation angle, so the azimuth only fixes a *vertical half-plane*
through the transmitter (the plane containing the Tx ellipsoid normal):

    P(s, t) = Tx + s * h + t * u        s >= 0

with h the horizontal unit vector at the target azimuth and u the local "up"
at the Tx.  Inside that half-plane the altitude constraint gives a curve
t = t(s) (the upper crossing of the constant-height surface), and the bistatic
range constraint

    |P - Tx| + |P - Rx| = |Rx - Tx| + excess_range

becomes a scalar equation g(s) = 0 along that curve.  Roots are bracketed on a
fine grid in s and polished by bisection.  No root -> no answer.

Public API
----------
bistatic_to_ecef(...)      -> np.ndarray (3,) or None
bistatic_to_ecef_all(...)  -> list of np.ndarray (all valid solutions, may be empty)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# WGS-84
# --------------------------------------------------------------------------- #
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
_MEAN_R = 6371008.8  # sphere radius used only for initial guesses


def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """Geodetic (rad, rad, m above ellipsoid) -> ECEF [m]."""
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array(
        [
            (n + alt_m) * cl * np.cos(lon_rad),
            (n + alt_m) * cl * np.sin(lon_rad),
            (n * (1.0 - WGS84_E2) + alt_m) * sl,
        ]
    )


def ecef_to_geodetic(ecef):
    """ECEF [m] (..., 3) -> (lat_rad, lon_rad, height_m), vectorised."""
    p = np.asarray(ecef, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rho = np.hypot(x, y)
    lat = np.arctan2(z, rho * (1.0 - WGS84_E2))
    for _ in range(6):
        sl = np.sin(lat)
        w = np.sqrt(1.0 - WGS84_E2 * sl * sl)
        n = WGS84_A / w
        h = rho * np.cos(lat) + z * sl - WGS84_A * w  # robust near the poles
        lat = np.arctan2(z, rho * (1.0 - WGS84_E2 * n / (n + h)))
    sl, cl = np.sin(lat), np.cos(lat)
    h = rho * cl + z * sl - WGS84_A * np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return lat, lon, h


def _enu_basis(lat: float, lon: float):
    """East, north, up unit vectors (in ECEF) at a geodetic lat/lon."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return east, north, up


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #
def bistatic_to_ecef_all(
    tx_ecef,
    rx_ecef,
    excess_range: float,
    azimuth: float,
    altitude: float,
    *,
    n_grid: int = 4000,
    tol: float = 1e-9,
) -> List[np.ndarray]:
    """
    Return every ECEF position consistent with the bistatic measurement,
    ordered by increasing horizontal distance from the transmitter.
    An empty list means the measurement is not physically realisable.

    Parameters
    ----------
    tx_ecef, rx_ecef : (3,) ECEF positions [m]
    excess_range     : (|T-Tx| + |T-Rx|) - |Rx-Tx|  [m]
    azimuth          : at Tx, clockwise from the Tx->Rx line of sight [rad]
    altitude         : target height above the WGS-84 ellipsoid [m]
    n_grid           : grid points used to bracket roots
    tol              : convergence tolerance [m]
    """
    tx = np.asarray(tx_ecef, dtype=float).reshape(3)
    rx = np.asarray(rx_ecef, dtype=float).reshape(3)

    # ---- basic sanity checks ------------------------------------------------
    if not (np.all(np.isfinite(tx)) and np.all(np.isfinite(rx))):
        return []
    if not np.all(np.isfinite([excess_range, azimuth, altitude])):
        return []
    if excess_range < 0.0:
        return []
    if _MEAN_R + altitude <= 0.0:
        return []

    lat0, lon0, h_tx = ecef_to_geodetic(tx)
    east, north, up = _enu_basis(lat0, lon0)

    d = rx - tx
    baseline = float(np.linalg.norm(d))
    e_rx, n_rx = float(d @ east), float(d @ north)
    if np.hypot(e_rx, n_rx) < 1e-3:
        return []  # Rx straight above/below Tx: azimuth reference undefined

    # ---- azimuth -> horizontal direction of the half-plane ------------------
    bearing = np.arctan2(e_rx, n_rx) + azimuth  # compass bearing, clockwise
    h_dir = np.sin(bearing) * east + np.cos(bearing) * north

    range_sum = baseline + excess_range
    # d1 <= (sum + baseline)/2 by the triangle inequality, and s <= d1
    s_max = 0.5 * (range_sum + baseline)

    # ---- altitude constraint: point on the half-plane at the given height ---
    def arc_points(s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        with np.errstate(all="ignore"):
            root = (_MEAN_R + altitude) ** 2 - s**2
            t = np.where(
                root > 0.0,
                -(_MEAN_R + h_tx) + np.sqrt(np.where(root > 0.0, root, 0.0)),
                np.nan,
            )
            dt = np.full(s.shape, np.inf)
            for _ in range(40):
                pts = tx + s[:, None] * h_dir + t[:, None] * up
                lat, lon, h = ecef_to_geodetic(pts)
                normal = np.stack(
                    [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
                    axis=-1,
                )
                slope = normal @ up
                t = np.where(slope > 1e-3, t, np.nan)  # grazing: ill-conditioned
                dt = (altitude - h) / slope
                t = t + dt
                finite = np.isfinite(dt)
                if not finite.any() or np.max(np.abs(dt[finite])) < tol:
                    break
            t = np.where(np.abs(dt) < 1e-6, t, np.nan)
            pts = tx + s[:, None] * h_dir + t[:, None] * up
        return pts

    def g_of(pts):
        return (
            np.linalg.norm(pts - tx, axis=-1)
            + np.linalg.norm(pts - rx, axis=-1)
            - range_sum
        )

    # ---- bracket roots of g(s) on a grid, then bisect -----------------------
    s_grid = np.linspace(0.0, s_max, n_grid)
    g = g_of(arc_points(s_grid))

    roots = [s_grid[i] for i in np.flatnonzero(g == 0.0)]
    with np.errstate(invalid="ignore"):
        brackets = np.flatnonzero(g[:-1] * g[1:] < 0.0)
    for i in brackets:
        lo, hi, g_lo = s_grid[i], s_grid[i + 1], g[i]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            g_mid = g_of(arc_points(mid))[0]
            if not np.isfinite(g_mid):
                break
            if g_lo * g_mid <= 0.0:
                hi = mid
            else:
                lo, g_lo = mid, g_mid
            if hi - lo < tol:
                break
        roots.append(0.5 * (lo + hi))

    solutions: List[np.ndarray] = []
    last_s = -np.inf
    for s in sorted(roots):
        if s - last_s < 1e-6:
            continue
        last_s = s
        solutions.append(arc_points(s)[0])
    return solutions


def bistatic_to_ecef(
    tx_ecef,
    rx_ecef,
    excess_range: float,
    azimuth: float,
    altitude: float,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Convert one bistatic measurement to a target ECEF position [m].

    Returns None if no position satisfies the excess range, azimuth and altitude
    simultaneously.  If more than one position does (possible for targets close
    to the Tx-Rx baseline), the one nearest the transmitter is returned; use
    `bistatic_to_ecef_all` to get every candidate.
    """
    solutions = bistatic_to_ecef_all(
        tx_ecef, rx_ecef, excess_range, azimuth, altitude, **kwargs
    )
    return solutions[0] if solutions else None
