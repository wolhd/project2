"""
Extract / interpolate aircraft track points at arbitrary query times.

Approach:
  1. Convert each track point (lat, lon, alt) to ECEF (x, y, z) -- interpolating
     in a Cartesian frame avoids the discontinuities / distortion you get
     interpolating lat/lon directly (antimeridian, poles, non-linear lon scaling).
  2. Per aircraft, fit a cubic spline of x(t), y(t), z(t).
  3. Evaluate the spline at the requested query times (only within that
     aircraft's own track time range -- no extrapolation).
  4. Convert the interpolated ECEF points back to lat/lon/alt.

Requires: numpy, scipy, pandas, pymap3d
"""

import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.interpolate import CubicSpline


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """Geodetic (lat/lon in degrees, alt in meters, WGS84) -> ECEF (x, y, z) in
    meters. Thin vectorized wrapper around pymap3d.geodetic2ecef."""
    x, y, z = pm.geodetic2ecef(
        np.asarray(lat_deg, dtype=float),
        np.asarray(lon_deg, dtype=float),
        np.asarray(alt_m, dtype=float),
    )
    return x, y, z


def ecef_to_lla(x, y, z):
    """ECEF (meters) -> geodetic (lat_deg, lon_deg, alt_m), WGS84.
    Thin vectorized wrapper around pymap3d.ecef2geodetic."""
    lat, lon, alt = pm.ecef2geodetic(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    return lat, lon, alt


# ---------------------------------------------------------------------------
# Core extraction / interpolation
# ---------------------------------------------------------------------------
def extract_track_points(
    df,
    query_times,
    id_col="aircraft_id",
    time_col="time",
    lat_col="lat",
    lon_col="lon",
    alt_col="alt",
    allow_extrapolate=False,
):
    """
    df: DataFrame with columns [id_col, time_col, lat_col, lon_col, alt_col].
        time_col must be numeric (e.g. unix seconds) or datetime64 -- both work,
        sorting/spline math just needs a monotonic numeric axis, which we build
        internally.
    query_times: 1D array-like of the times you want track points at (the
        "window" of time points). Same dtype family as df[time_col]
        (numeric or datetime64).
    allow_extrapolate: if False (default), query times that fall outside an
        aircraft's own [first_point_time, last_point_time] range are dropped
        for that aircraft rather than extrapolated.

    Returns a long DataFrame: [id_col, time_col, lat, lon, alt, interpolated]
    'interpolated' is False when the query time exactly matches an original
    sample time (to float tolerance) that aircraft actually reported, True
    otherwise.
    """
    query_times = pd.Series(query_times)
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[time_col])

    # Work in float seconds internally regardless of whether time is
    # numeric or a datetime64 column.
    def to_float_seconds(series):
        if is_datetime:
            return series.astype("int64") / 1e9  # ns -> s
        return series.astype(float)

    df = df.copy()
    df["_t"] = to_float_seconds(df[time_col])
    q_t = to_float_seconds(query_times).to_numpy()

    results = []

    for ac_id, g in df.groupby(id_col, sort=False):
        g = g.sort_values("_t")
        g = g.drop_duplicates(subset="_t", keep="first")
        t = g["_t"].to_numpy()

        if len(t) < 2:
            continue  # nothing to interpolate against

        # ECEF for this aircraft's track
        x, y, z = lla_to_ecef(g[lat_col].to_numpy(), g[lon_col].to_numpy(), g[alt_col].to_numpy())

        # 'natural' boundary condition works for as few as 2 points;
        # not-a-knot (scipy default) needs >= 4, so pick automatically.
        bc_type = "not-a-knot" if len(t) >= 4 else "natural"
        cs_x = CubicSpline(t, x, bc_type=bc_type)
        cs_y = CubicSpline(t, y, bc_type=bc_type)
        cs_z = CubicSpline(t, z, bc_type=bc_type)

        if allow_extrapolate:
            mask = np.ones_like(q_t, dtype=bool)
        else:
            mask = (q_t >= t.min()) & (q_t <= t.max())

        qt = q_t[mask]
        if len(qt) == 0:
            continue

        xi, yi, zi = cs_x(qt), cs_y(qt), cs_z(qt)
        lat_i, lon_i, alt_i = ecef_to_lla(xi, yi, zi)

        # Flag points that coincide with an original sample (within 1e-6 s)
        was_sampled = np.isin(np.round(qt, 6), np.round(t, 6))

        out = pd.DataFrame(
            {
                id_col: ac_id,
                time_col: query_times[mask].to_numpy(),
                "lat": lat_i,
                "lon": lon_i,
                "alt": alt_i,
                "interpolated": ~was_sampled,
            }
        )
        results.append(out)

    if not results:
        return pd.DataFrame(columns=[id_col, time_col, "lat", "lon", "alt", "interpolated"])

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Quick self-test / example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Two synthetic aircraft, ~1 sample every 10s over a few minutes
    rows = []
    for ac, (lat0, lon0, alt0) in {
        "AC1": (42.0, -71.0, 10000.0),
        "AC2": (42.2, -70.8, 11000.0),
    }.items():
        t0 = 0
        for i in range(20):
            t = t0 + i * 10
            rows.append(
                dict(
                    aircraft_id=ac,
                    time=t,
                    lat=lat0 + 0.001 * i + rng.normal(0, 1e-5),
                    lon=lon0 - 0.0008 * i + rng.normal(0, 1e-5),
                    alt=alt0 + 5 * i,
                )
            )
    df = pd.DataFrame(rows)

    # Round-trip sanity check on the ECEF conversion itself
    x, y, z = lla_to_ecef(df["lat"], df["lon"], df["alt"])
    lat2, lon2, alt2 = ecef_to_lla(x, y, z)
    assert np.allclose(lat2, df["lat"], atol=1e-9)
    assert np.allclose(lon2, df["lon"], atol=1e-9)
    assert np.allclose(alt2, df["alt"], atol=1e-6)
    print("ECEF round-trip OK")

    # A window of query times, some matching samples, some in between,
    # and a couple outside AC1's range to prove those get dropped (not extrapolated)
    window = [5, 15, 22.5, 100, 195, 250]
    out = extract_track_points(df, window)
    print(out.to_string(index=False))
