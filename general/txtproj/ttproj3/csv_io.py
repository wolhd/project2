"""CSV readers.  Headers are case-insensitive; blank lines and ``#`` lines are skipped.

detections.csv  (required unless marked optional)
    time_s, range_m, range_rate_mps, range_std_m, range_rate_std_mps,
    rx_x_m, rx_y_m, rx_z_m, rx_vx_mps, rx_vy_mps, rx_vz_mps,
    and EITHER azimuth_deg + azimuth_std_deg  OR  azimuth_rad + azimuth_std_rad

known_aircraft.csv
    time_s, icao, and position as EITHER x_m, y_m, z_m (ECEF)
                                  OR     lat_deg, lon_deg, alt_m (WGS-84 ellipsoidal height)
    optional velocity as EITHER vx_mps, vy_mps, vz_mps (ECEF)
                         OR     ve_mps, vn_mps, vu_mps (ENU; needs geodetic position)
    Blank velocity cells mean "unknown".

tx_hint.csv
    time_s, position as above (x_m.. or lat_deg..), optional pos_std_m (1-sigma, blank = unknown)

ADS-B reports barometric altitude and geometric (GNSS) altitude separately;
``alt_m`` here must be height above the WGS-84 ellipsoid (geometric), not baro.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

from .geodesy import enu_to_ecef_rotation, geodetic_to_ecef
from .models import (
    BistaticDetection,
    DetectionSequence,
    KnownAircraftSequence,
    KnownAircraftState,
    TxHintSequence,
    TxLocationHint,
)
from .validation import FloatArray

StrPath = str | os.PathLike[str]
_Row = dict[str, str]

_DET_REQUIRED = (
    "time_s", "range_m", "range_rate_mps", "range_std_m", "range_rate_std_mps",
    "rx_x_m", "rx_y_m", "rx_z_m", "rx_vx_mps", "rx_vy_mps", "rx_vz_mps",
)  # fmt: skip
_ECEF_POS = ("x_m", "y_m", "z_m")
_GEO_POS = ("lat_deg", "lon_deg", "alt_m")
_ECEF_VEL = ("vx_mps", "vy_mps", "vz_mps")
_ENU_VEL = ("ve_mps", "vn_mps", "vu_mps")


class CsvFormatError(ValueError):
    """Raised for malformed CSV structure or content (message includes file and line)."""


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _load_table(path: StrPath) -> tuple[frozenset[str], list[tuple[int, _Row]]]:
    p = Path(path)
    header: list[str] | None = None
    rows: list[tuple[int, _Row]] = []
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for raw in reader:
            if not any(c.strip() for c in raw) or raw[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [c.strip().lower() for c in raw]
                if len(set(header)) != len(header):
                    raise CsvFormatError(f"{p}: duplicate column names in header")
                continue
            rows.append((reader.line_num, {h: v.strip() for h, v in zip(header, raw)}))
    if header is None:
        raise CsvFormatError(f"{p}: file has no header row")
    return frozenset(header), rows


def _require(cols: frozenset[str], required: tuple[str, ...], path: StrPath) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise CsvFormatError(f"{path}: missing required column(s): {', '.join(missing)}")


def _f(row: _Row, key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        raise ValueError(f"missing value for column {key!r}")
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"column {key!r}: cannot parse {raw!r} as a number") from None


def _optional_triplet(row: _Row, keys: tuple[str, ...]) -> tuple[float, float, float] | None:
    present = [row.get(k, "") != "" for k in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"columns {keys} must be all filled or all blank")
    return _f(row, keys[0]), _f(row, keys[1]), _f(row, keys[2])


def _is_geodetic(cols: frozenset[str], path: StrPath) -> bool:
    has_ecef, has_geo = set(_ECEF_POS) <= cols, set(_GEO_POS) <= cols
    if has_ecef == has_geo:
        raise CsvFormatError(
            f"{path}: need exactly one position format: ECEF {_ECEF_POS} or geodetic {_GEO_POS}"
        )
    return has_geo


def _velocity_format(cols: frozenset[str], geodetic: bool, path: StrPath) -> str | None:
    has_ecef, has_enu = set(_ECEF_VEL) <= cols, set(_ENU_VEL) <= cols
    if has_ecef and has_enu:
        raise CsvFormatError(f"{path}: give velocity as ECEF or ENU columns, not both")
    if has_enu and not geodetic:
        raise CsvFormatError(f"{path}: ENU velocity columns require geodetic position columns")
    return "ecef" if has_ecef else "enu" if has_enu else None


def _position(row: _Row, geodetic: bool) -> tuple[FloatArray, tuple[float, float] | None]:
    if geodetic:
        lat, lon, alt = (_f(row, k) for k in _GEO_POS)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0):
            raise ValueError(f"lat/lon out of range: {lat}, {lon}")
        return geodetic_to_ecef(lat, lon, alt), (lat, lon)
    return np.array([_f(row, k) for k in _ECEF_POS], dtype=np.float64), None


# --------------------------------------------------------------------------- #
# Public readers
# --------------------------------------------------------------------------- #
def read_detections_csv(path: StrPath) -> DetectionSequence:
    """Read bistatic detections; azimuth columns may be in degrees or radians."""
    cols, rows = _load_table(path)
    _require(cols, _DET_REQUIRED, path)
    has_deg = {"azimuth_deg", "azimuth_std_deg"} <= cols
    has_rad = {"azimuth_rad", "azimuth_std_rad"} <= cols
    if has_deg == has_rad:
        raise CsvFormatError(
            f"{path}: need exactly one azimuth unit: azimuth_deg+azimuth_std_deg "
            "or azimuth_rad+azimuth_std_rad"
        )
    unit = "deg" if has_deg else "rad"
    scale = math.radians(1.0) if has_deg else 1.0

    out: list[BistaticDetection] = []
    for line, row in rows:
        try:
            out.append(
                BistaticDetection(
                    time_s=_f(row, "time_s"),
                    range_m=_f(row, "range_m"),
                    azimuth_rad=_f(row, f"azimuth_{unit}") * scale,
                    range_rate_mps=_f(row, "range_rate_mps"),
                    range_std_m=_f(row, "range_std_m"),
                    azimuth_std_rad=_f(row, f"azimuth_std_{unit}") * scale,
                    range_rate_std_mps=_f(row, "range_rate_std_mps"),
                    rx_pos_ecef_m=np.array([_f(row, k) for k in ("rx_x_m", "rx_y_m", "rx_z_m")]),
                    rx_vel_ecef_mps=np.array(
                        [_f(row, k) for k in ("rx_vx_mps", "rx_vy_mps", "rx_vz_mps")]
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CsvFormatError(f"{path}: line {line}: {exc}") from exc
    return DetectionSequence(out)


def read_known_aircraft_csv(path: StrPath) -> KnownAircraftSequence:
    """Read known-aircraft (ADS-B) states; see module docstring for columns."""
    cols, rows = _load_table(path)
    _require(cols, ("time_s", "icao"), path)
    geodetic = _is_geodetic(cols, path)
    vel_fmt = _velocity_format(cols, geodetic, path)

    out: list[KnownAircraftState] = []
    for line, row in rows:
        try:
            pos, lat_lon = _position(row, geodetic)
            vel: FloatArray | None = None
            if vel_fmt == "ecef":
                triplet = _optional_triplet(row, _ECEF_VEL)
                vel = None if triplet is None else np.array(triplet, dtype=np.float64)
            elif vel_fmt == "enu" and lat_lon is not None:
                triplet = _optional_triplet(row, _ENU_VEL)
                if triplet is not None:
                    vel = enu_to_ecef_rotation(*lat_lon) @ np.array(triplet, dtype=np.float64)
            out.append(
                KnownAircraftState(
                    time_s=_f(row, "time_s"),
                    icao=row.get("icao", ""),
                    pos_ecef_m=pos,
                    vel_ecef_mps=vel,
                )
            )
        except (TypeError, ValueError) as exc:
            raise CsvFormatError(f"{path}: line {line}: {exc}") from exc
    return KnownAircraftSequence(out)


def read_tx_hint_csv(path: StrPath) -> TxHintSequence:
    """Read suggested transmitter locations; optional ``pos_std_m`` column."""
    cols, rows = _load_table(path)
    _require(cols, ("time_s",), path)
    geodetic = _is_geodetic(cols, path)

    out: list[TxLocationHint] = []
    for line, row in rows:
        try:
            pos, _ = _position(row, geodetic)
            std = _f(row, "pos_std_m") if row.get("pos_std_m", "") != "" else None
            out.append(TxLocationHint(time_s=_f(row, "time_s"), pos_ecef_m=pos, pos_std_m=std))
        except (TypeError, ValueError) as exc:
            raise CsvFormatError(f"{path}: line {line}: {exc}") from exc
    return TxHintSequence(out)
