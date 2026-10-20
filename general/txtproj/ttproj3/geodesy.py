"""Minimal WGS-84 helpers (only what the CSV readers need)."""

from __future__ import annotations

import math
from typing import Final, NamedTuple

import numpy as np

from .validation import FloatArray

WGS84_A: Final[float] = 6_378_137.0  # semi-major axis [m]
WGS84_F: Final[float] = 1.0 / 298.257223563  # flattening
WGS84_E2: Final[float] = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> FloatArray:
    """Geodetic lat/lon/height above the WGS-84 ellipsoid -> ECEF [m]."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    return np.array(
        [
            (n + alt_m) * cos_lat * np.cos(lon),
            (n + alt_m) * cos_lat * np.sin(lon),
            (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> FloatArray:
    """3x3 matrix ``R`` such that ``v_ecef = R @ v_enu`` at the given lat/lon."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    east = [-so, co, 0.0]
    north = [-sl * co, -sl * so, cl]
    up = [cl * co, cl * so, sl]
    return np.column_stack((east, north, up)).astype(np.float64)


def ecef_to_geodetic(ecef_m: FloatArray) -> tuple[float, float, float]:
    """ECEF [m] -> (lat_deg, lon_deg, ellipsoidal height m); fixed-point iteration."""
    x, y, z = (float(c) for c in ecef_m)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(12):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        if abs(lat) < math.radians(80.0):
            alt = p / math.cos(lat) - n
        else:  # avoid dividing by cos(lat) near the poles
            alt = z / sin_lat - n * (1.0 - WGS84_E2)
        new_lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
        converged = abs(new_lat - lat) < 1e-13
        lat = new_lat
        if converged:
            break
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n if abs(lat) < math.radians(80.0) else z / sin_lat - n * (1.0 - WGS84_E2)
    return math.degrees(lat), math.degrees(lon), alt


def wrap_pi(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class EllipsoidFrame(NamedTuple):
    """Local ENU axes (unit vectors, ECEF components) at a point, plus d(up)/d(point)."""

    east: FloatArray
    north: FloatArray
    up: FloatArray  # WGS-84 ellipsoid normal (geodetic vertical)
    up_jacobian: FloatArray  # (3, 3), symmetric: how ``up`` changes as the point moves


def ellipsoid_frame(ecef_m: FloatArray) -> EllipsoidFrame:
    """Local frame at an ECEF point.

    ``up_jacobian = N N^T/(M+h) + E E^T/(R_n+h)`` with M, R_n the meridian and
    prime-vertical radii of curvature and h the ellipsoidal height: moving the
    point north by ds tilts ``up`` towards north by ds/(M+h), and likewise east.
    """
    lat_deg, lon_deg, alt = ecef_to_geodetic(ecef_m)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    w = math.sqrt(1.0 - WGS84_E2 * sl * sl)
    r_n = WGS84_A / w
    r_m = WGS84_A * (1.0 - WGS84_E2) / (w * w * w)
    jac = np.outer(north, north) / (r_m + alt) + np.outer(east, east) / (r_n + alt)
    return EllipsoidFrame(east, north, up, jac)
