from trackerapp.models.types import GeodeticPosition
from trackerapp.transforms.geodetic_ecef import ecef_to_geodetic, geodetic_to_ecef


def test_round_trip_equator():
    original = GeodeticPosition(
        track_id="t1", latitude_deg=0.0, longitude_deg=0.0, altitude_m=0.0, timestamp=100.0
    )
    ecef = geodetic_to_ecef(original)
    # on the equator/prime meridian, x should equal ~ WGS84 semi-major axis
    assert abs(ecef.x_m - 6378137.0) < 1e-3
    assert abs(ecef.y_m) < 1e-6
    assert abs(ecef.z_m) < 1e-6

    back = ecef_to_geodetic(ecef)
    assert abs(back.latitude_deg - original.latitude_deg) < 1e-9
    assert abs(back.longitude_deg - original.longitude_deg) < 1e-9
    assert abs(back.altitude_m - original.altitude_m) < 1e-6


def test_round_trip_arbitrary_point():
    original = GeodeticPosition(
        track_id="t2",
        latitude_deg=42.3601,
        longitude_deg=-71.0589,
        altitude_m=15.0,
        timestamp=200.0,
    )
    ecef = geodetic_to_ecef(original)
    back = ecef_to_geodetic(ecef)
    assert abs(back.latitude_deg - original.latitude_deg) < 1e-9
    assert abs(back.longitude_deg - original.longitude_deg) < 1e-9
    assert abs(back.altitude_m - original.altitude_m) < 1e-6
    assert back.track_id == original.track_id
