"""Geodetic (WGS84 lat/lon/alt) <-> ECEF coordinate transforms.

Pure-function module, no external dependencies beyond the stdlib `math`, so
it stays trivially testable. If higher precision or datum flexibility is
ever needed, swap the body of these two functions for a call into `pymap3d`
or similar without touching any call sites (both take/return the same
dataclasses).
"""

from __future__ import annotations

import math

from trackerapp.models.types import EcefPosition, GeodeticPosition

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0  # semi-major axis, meters
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)  # first eccentricity squared


def geodetic_to_ecef(pos: GeodeticPosition) -> EcefPosition:
    """Convert a geodetic position (degrees, meters) to ECEF (meters)."""
    lat_rad = math.radians(pos.latitude_deg)
    lon_rad = math.radians(pos.longitude_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # prime vertical radius of curvature
    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat * sin_lat)

    x = (n + pos.altitude_m) * cos_lat * math.cos(lon_rad)
    y = (n + pos.altitude_m) * cos_lat * math.sin(lon_rad)
    z = (n * (1 - _WGS84_E2) + pos.altitude_m) * sin_lat

    return EcefPosition(
        track_id=pos.track_id,
        x_m=x,
        y_m=y,
        z_m=z,
        timestamp=pos.timestamp,
    )


def ecef_to_geodetic(pos: EcefPosition) -> GeodeticPosition:
    """Convert an ECEF position (meters) back to geodetic (degrees, meters).

    Uses the iterative Bowring method; converges to sub-millimeter accuracy
    in a handful of iterations for any position near the earth's surface.
    """
    x, y, z = pos.x_m, pos.y_m, pos.z_m
    lon_rad = math.atan2(y, x)

    p = math.hypot(x, y)
    lat_rad = math.atan2(z, p * (1 - _WGS84_E2))  # initial guess

    for _ in range(5):
        sin_lat = math.sin(lat_rad)
        n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat * sin_lat)
        alt_m = p / math.cos(lat_rad) - n
        lat_rad = math.atan2(z, p * (1 - _WGS84_E2 * n / (n + alt_m)))

    sin_lat = math.sin(lat_rad)
    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat * sin_lat)
    alt_m = p / math.cos(lat_rad) - n

    return GeodeticPosition(
        track_id=pos.track_id,
        latitude_deg=math.degrees(lat_rad),
        longitude_deg=math.degrees(lon_rad),
        altitude_m=alt_m,
        timestamp=pos.timestamp,
    )
