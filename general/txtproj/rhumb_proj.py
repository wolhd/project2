from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta

import pymap3d as pm


@dataclass(frozen=True)
class Track:
    """
    Position and velocity of a moving point.

    lat, lon      : degrees (WGS84)
    alt           : meters, ellipsoidal
    heading       : degrees true, clockwise from north (the constant rhumb bearing)
    speed         : meters/second, ground speed along the surface
    climb_rate    : meters/second, vertical speed (default 0, constant altitude)
    """
    lat: float
    lon: float
    alt: float
    heading: float
    speed: float
    climb_rate: float = 0.0

    def project(self, dt: float | timedelta) -> Track:
        """
        Return a new Track after dt (seconds or a timedelta), moving along a
        rhumb line (constant heading) at constant speed. Negative dt projects
        backward. Heading, speed and climb rate are unchanged.
        """
        if isinstance(dt, timedelta):
            dt = dt.total_seconds()

        dist = self.speed * dt
        lat2, lon2 = pm.loxodrome_direct(
            self.lat, self.lon, dist, self.heading, deg=True
        )
        lon2 = (lon2 + 180.0) % 360.0 - 180.0   # wrap to [-180, 180)

        return replace(
            self,
            lat=float(lat2),
            lon=float(lon2),
            alt=self.alt + self.climb_rate * dt,
        )


if __name__ == "__main__":
    t0 = Track(lat=42.6334, lon=-71.3162, alt=30.0, heading=45.0, speed=15.0)
    t1 = t0.project(600)                       # 10 minutes
    print(t1)
    print(t0.project(timedelta(minutes=10)) == t1)
