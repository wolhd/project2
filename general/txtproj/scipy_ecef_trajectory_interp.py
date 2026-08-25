"""
scipy_ecef_trajectory_interp.py

Interpolate / extrapolate (lat, lon, altitude, time) tracks using
scipy.interpolate spline fitting, done directly in ECEF (Earth-Centered,
Earth-Fixed) Cartesian coordinates.

Why ECEF and not lat/lon directly
----------------------------------
Interpolating lat/lon degrees directly breaks down near the poles and
across the +/-180 longitude seam, and isn't physically a straight-line
path in 3D. ECEF is a single global Cartesian frame (meters, origin at
Earth's center), so splining x(t), y(t), z(t) independently is a
geometrically sane, seam-free way to fit a smooth 3D curve through the
points -- no local tangent-plane (ENU) construction needed, at the cost
of the polynomial having to arc very slightly with Earth's curvature
over long distances (usually negligible for tracks under a few hundred
km).

Interpolation uses a cubic spline through x(t), y(t), z(t). Extrapolation
beyond the first/last timestamps is done *linearly*, using the spline's
boundary slope (i.e. constant velocity from the edge) rather than letting
the cubic polynomial run off unconstrained -- cubic extrapolation is
notoriously unstable/wild outside its fitted range.

Requires numpy and scipy.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

# ----------------------------------------------------------------------
# WGS84 <-> ECEF (same geodesy as the Kalman version, no ENU step needed)
# ----------------------------------------------------------------------

_A = 6378137.0                       # semi-major axis (m)
_F = 1 / 298.257223563               # flattening
_B = _A * (1 - _F)                   # semi-minor axis (m)
_E2 = 1 - (_B ** 2) / (_A ** 2)      # first eccentricity squared
_EP2 = (_A ** 2 - _B ** 2) / _B ** 2 # second eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - _E2) + alt_m) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * _A, p * _B)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + _EP2 * _B * np.sin(theta) ** 3,
        p - _E2 * _A * np.cos(theta) ** 3,
    )
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


# ----------------------------------------------------------------------
# Interpolator
# ----------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    extrapolated: np.ndarray  # bool mask: True if outside the measured span


class TrajectoryScipyInterpolator:
    def __init__(self, method: str = "cubic"):
        """
        method: 'cubic'  -> CubicSpline (smooth, can overshoot near sharp
                             turns/noisy points)
                'pchip'  -> PchipInterpolator (shape-preserving/monotonic
                             per-axis, won't overshoot -- safer with noisy
                             real-world GPS data)
        """
        if method not in ("cubic", "pchip"):
            raise ValueError("method must be 'cubic' or 'pchip'")
        self.method = method
        self._fitted = False

    def fit(self, times, lats, lons, alts):
        times = np.asarray(times, dtype=float)
        order = np.argsort(times)
        times = times[order]
        lats = np.asarray(lats, dtype=float)[order]
        lons = np.asarray(lons, dtype=float)[order]
        alts = np.asarray(alts, dtype=float)[order]
        if len(times) < 2:
            raise ValueError("Need at least 2 points to fit a trajectory.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Duplicate timestamps after sorting; times must be strictly increasing.")

        ecef = geodetic_to_ecef(lats, lons, alts)  # (N, 3)

        Interp = CubicSpline if self.method == "cubic" else PchipInterpolator
        # extrapolate=False -> raises/NaNs outside range; we handle
        # extrapolation ourselves (linearly) below instead.
        self._splines = [Interp(times, ecef[:, i], extrapolate=False) for i in range(3)]

        self.t_min, self.t_max = times[0], times[-1]
        self._ecef_min = ecef[0]
        self._ecef_max = ecef[-1]
        # boundary velocities (m per time-unit), for linear extrapolation
        self._vel_min = np.array([s.derivative()(self.t_min) for s in self._splines])
        self._vel_max = np.array([s.derivative()(self.t_max) for s in self._splines])
        self._fitted = True
        return self

    def predict(self, query_times) -> TrajectoryResult:
        if not self._fitted:
            raise RuntimeError("Call .fit(...) before .predict(...)")

        query_times = np.asarray(query_times, dtype=float)
        before = query_times < self.t_min
        after = query_times > self.t_max
        in_span = ~before & ~after

        ecef = np.zeros((len(query_times), 3))

        if np.any(in_span):
            for i, s in enumerate(self._splines):
                ecef[in_span, i] = s(query_times[in_span])

        if np.any(before):
            dt = (query_times[before] - self.t_min)[:, None]      # negative
            ecef[before] = self._ecef_min[None, :] + dt * self._vel_min[None, :]

        if np.any(after):
            dt = (query_times[after] - self.t_max)[:, None]       # positive
            ecef[after] = self._ecef_max[None, :] + dt * self._vel_max[None, :]

        lat, lon, alt = ecef_to_geodetic(ecef)
        return TrajectoryResult(
            time=query_times, lat=lat, lon=lon, alt=alt,
            extrapolated=before | after,
        )


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    times = [0, 12, 25, 41, 58, 75]
    lats  = [42.3601, 42.3607, 42.3614, 42.3625, 42.3639, 42.3651]
    lons  = [-71.0589, -71.0575, -71.0558, -71.0541, -71.0522, -71.0503]
    alts  = [10.0, 12.5, 11.0, 14.0, 13.5, 16.0]

    interp = TrajectoryScipyInterpolator(method="pchip")
    interp.fit(times, lats, lons, alts)

    query = np.arange(0, 95, 1.0)
    result = interp.predict(query)

    print(f"{'t':>6} {'lat':>12} {'lon':>12} {'alt':>8}  {'extrap'}")
    for i in range(0, len(query), 5):
        r = result
        print(f"{r.time[i]:6.1f} {r.lat[i]:12.6f} {r.lon[i]:12.6f} "
              f"{r.alt[i]:8.2f}  {'yes' if r.extrapolated[i] else ''}")
