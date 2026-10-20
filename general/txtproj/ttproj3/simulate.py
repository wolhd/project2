"""Synthetic scenario: static transmitter, moving receiver, known + unknown aircraft, clutter.

Unknown aircraft are real reflectors, so their detections are perfectly
consistent with the true transmitter; the estimator just does not know where
those aircraft are.  Clutter detections are random.  Truth labels are kept so a
run can be scored, but they never reach the estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geodesy import enu_to_ecef_rotation, geodetic_to_ecef
from .kinematics import KinematicState
from .measurement_model import BistaticMeasurementModel
from .models import (
    BistaticDetection,
    DetectionSequence,
    KnownAircraftSequence,
    KnownAircraftState,
    TxLocationHint,
)
from .validation import FloatArray, as_positive_float

CLUTTER = "CLUTTER"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    seed: int = 7
    origin_lat_deg: float = 40.0
    origin_lon_deg: float = -100.0
    duration_s: float = 120.0
    scan_period_s: float = 2.0
    adsb_period_s: float = 4.0
    tx_enu_m: tuple[float, float, float] = (-60e3, 25e3, 200.0)
    rx_velocity_enu_mps: tuple[float, float, float] = (15.0, 0.0, 0.0)
    hint_error_m: float = 12e3  # horizontal offset between hint and truth
    hint_std_m: float = 15e3
    n_unknown_aircraft: int = 8
    p_detect: float = 0.85
    clutter_per_scan: float = 2.0
    range_std_m: float = 100.0
    range_rate_std_mps: float = 1.0
    azimuth_std_rad: float = math.radians(1.0)
    adsb_pos_std_m: float = 10.0
    adsb_vel_std_mps: float = 0.3

    def __post_init__(self) -> None:
        as_positive_float(self.duration_s, "duration_s")
        as_positive_float(self.scan_period_s, "scan_period_s")
        as_positive_float(self.adsb_period_s, "adsb_period_s")


@dataclass(frozen=True, eq=False)
class Scenario:
    tx_truth: KinematicState
    detections: DetectionSequence
    labels: dict[tuple[float, float], str]  # (time_s, range_m) -> ICAO | UNKNOWN | CLUTTER
    adsb: KnownAircraftSequence
    hint: TxLocationHint
    config: ScenarioConfig = field(default_factory=ScenarioConfig)


# Known aircraft: (icao, start ENU [m], velocity ENU [m/s])
_KNOWN = (
    ("A1B2C3", (-20e3, 60e3, 9e3), (220.0, 0.0, 0.0)),
    ("D4E5F6", (60e3, -30e3, 10e3), (-150.0, 120.0, 0.0)),
    ("0A1B2C", (10e3, 20e3, 8e3), (0.0, -180.0, 0.0)),
)


def make_scenario(config: ScenarioConfig | None = None) -> Scenario:
    cfg = config or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    origin = geodetic_to_ecef(cfg.origin_lat_deg, cfg.origin_lon_deg, 0.0)
    rot = enu_to_ecef_rotation(cfg.origin_lat_deg, cfg.origin_lon_deg)

    def pos(enu: tuple[float, float, float] | FloatArray) -> FloatArray:
        return origin + rot @ np.asarray(enu, dtype=np.float64)

    def vel(enu: tuple[float, float, float] | FloatArray) -> FloatArray:
        return rot @ np.asarray(enu, dtype=np.float64)

    model = BistaticMeasurementModel()
    tx = KinematicState(pos(cfg.tx_enu_m), np.zeros(3))
    rx_vel = vel(cfg.rx_velocity_enu_mps)

    # Aircraft: known (reported via ADS-B) and unknown (never reported).
    craft: list[tuple[str, FloatArray, FloatArray]] = [
        (icao, np.array(p, dtype=float), np.array(v, dtype=float)) for icao, p, v in _KNOWN
    ]
    for k in range(cfg.n_unknown_aircraft):
        heading = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(100.0, 250.0)
        start = [rng.uniform(-80e3, 80e3), rng.uniform(-80e3, 80e3), rng.uniform(3e3, 12e3)]
        craft.append((f"{UNKNOWN}{k}", np.array(start), np.array([speed * math.sin(heading), speed * math.cos(heading), 0.0])))
    known_ids = {icao for icao, _, _ in _KNOWN}

    # Detections
    detections: list[BistaticDetection] = []
    labels: dict[tuple[float, float], str] = {}
    for t in np.arange(0.0, cfg.duration_s + 1e-9, cfg.scan_period_s):
        rx = KinematicState(origin + rx_vel * t, rx_vel)
        for icao, p0, v0 in craft:
            if rng.random() > cfg.p_detect:
                continue
            target = KinematicState(pos(p0 + v0 * t), vel(v0))
            det = model.simulate_detection(
                float(t), tx, target, rx,
                range_std_m=cfg.range_std_m, azimuth_std_rad=cfg.azimuth_std_rad,
                range_rate_std_mps=cfg.range_rate_std_mps, rng=rng,
            )  # fmt: skip
            detections.append(det)
            labels[(det.time_s, det.range_m)] = icao if icao in known_ids else UNKNOWN
        for _ in range(int(rng.poisson(cfg.clutter_per_scan))):
            det = BistaticDetection(
                time_s=float(t),
                range_m=float(rng.uniform(0.0, 1e5)),
                azimuth_rad=float(rng.uniform(-math.pi, math.pi)),
                range_rate_mps=float(rng.uniform(-300.0, 300.0)),
                range_std_m=cfg.range_std_m,
                azimuth_std_rad=cfg.azimuth_std_rad,
                range_rate_std_mps=cfg.range_rate_std_mps,
                rx_pos_ecef_m=rx.pos_ecef_m,
                rx_vel_ecef_mps=rx.vel_ecef_mps,
            )
            detections.append(det)
            labels[(det.time_s, det.range_m)] = CLUTTER

    # ADS-B reports (known aircraft only), with measurement noise.
    reports: list[KnownAircraftState] = []
    for icao, p0, v0 in craft:
        if icao not in known_ids:
            continue
        for t in np.arange(0.0, cfg.duration_s + 1e-9, cfg.adsb_period_s):
            reports.append(
                KnownAircraftState(
                    float(t), icao,
                    pos(p0 + v0 * t) + rng.normal(0.0, cfg.adsb_pos_std_m, 3),
                    vel(v0) + rng.normal(0.0, cfg.adsb_vel_std_mps, 3),
                )
            )  # fmt: skip

    # Hint: truth displaced horizontally by hint_error_m in a random direction.
    ang = rng.uniform(0, 2 * math.pi)
    off = (cfg.hint_error_m * math.sin(ang), cfg.hint_error_m * math.cos(ang), 0.0)
    hint = TxLocationHint(0.0, pos(np.asarray(cfg.tx_enu_m) + off), cfg.hint_std_m)

    return Scenario(tx, DetectionSequence(detections), labels, KnownAircraftSequence(reports), hint, cfg)


def write_csvs(scenario: Scenario, folder: Path) -> tuple[Path, Path, Path]:
    """Write detections / ADS-B / tx-hint CSVs in the formats ``csv_io`` reads."""
    folder.mkdir(parents=True, exist_ok=True)
    det_path, adsb_path, hint_path = folder / "detections.csv", folder / "adsb.csv", folder / "tx_hint.csv"

    def f(*values: float) -> str:
        return ",".join(repr(float(v)) for v in values)  # repr -> exact round trip

    det_lines = [
        "time_s,range_m,azimuth_rad,range_rate_mps,range_std_m,azimuth_std_rad,range_rate_std_mps,"
        "rx_x_m,rx_y_m,rx_z_m,rx_vx_mps,rx_vy_mps,rx_vz_mps"
    ]
    for d in scenario.detections:
        det_lines.append(
            f(d.time_s, d.range_m, d.azimuth_rad, d.range_rate_mps, d.range_std_m, d.azimuth_std_rad,
              d.range_rate_std_mps, *d.rx_pos_ecef_m, *d.rx_vel_ecef_mps)
        )  # fmt: skip
    det_path.write_text("\n".join(det_lines) + "\n")

    adsb_lines = ["time_s,icao,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps"]
    for s in scenario.adsb:
        assert s.vel_ecef_mps is not None
        adsb_lines.append(f"{s.time_s!r},{s.icao}," + f(*s.pos_ecef_m, *s.vel_ecef_mps))
    adsb_path.write_text("\n".join(adsb_lines) + "\n")

    h = scenario.hint
    hint_path.write_text("time_s,x_m,y_m,z_m,pos_std_m\n" + f(h.time_s, *h.pos_ecef_m, h.pos_std_m or 0.0) + "\n")
    return det_path, adsb_path, hint_path
