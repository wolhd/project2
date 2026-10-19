from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from txloc import (
    BistaticDetection,
    CsvFormatError,
    DetectionSequence,
    KnownAircraftState,
    MeasurementBuffer,
    MeasurementPlayback,
    TxLocationHint,
    read_detections_csv,
    read_known_aircraft_csv,
    read_tx_hint_csv,
)
from txloc.geodesy import geodetic_to_ecef

RX = [1.0, 2.0, 3.0]


def det(t: float, **kw: object) -> BistaticDetection:
    args: dict[str, object] = dict(
        time_s=t, range_m=1e4, azimuth_rad=0.1, range_rate_mps=-5.0,
        range_std_m=50.0, azimuth_std_rad=0.02, range_rate_std_mps=1.0,
        rx_pos_ecef_m=RX, rx_vel_ecef_mps=[0.0, 0.0, 0.0],
    )  # fmt: skip
    args.update(kw)
    return BistaticDetection(**args)  # type: ignore[arg-type]


# ---- validation ---------------------------------------------------------- #
def test_detection_validates_and_freezes() -> None:
    d = det(1)  # ints accepted, normalised to float
    assert isinstance(d.time_s, float)
    assert not d.rx_pos_ecef_m.flags.writeable
    with pytest.raises(AttributeError):
        d.time_s = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kw", "exc"),
    [
        ({"range_std_m": 0.0}, ValueError),
        ({"range_m": float("nan")}, ValueError),
        ({"time_s": "1.0"}, TypeError),
        ({"time_s": True}, TypeError),
        ({"rx_pos_ecef_m": [1.0, 2.0]}, ValueError),
        ({"rx_pos_ecef_m": "abc"}, TypeError),
    ],
)
def test_detection_rejects_bad_input(kw: dict[str, object], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        det(0.0, **kw)


def test_sequence_rejects_wrong_item_type() -> None:
    with pytest.raises(TypeError):
        DetectionSequence([TxLocationHint(0.0, RX)])  # type: ignore[list-item]


def test_sequence_sorts_and_exposes_arrays() -> None:
    seq = DetectionSequence([det(3), det(1), det(2)])
    assert list(seq.times_s) == [1.0, 2.0, 3.0]
    assert seq.rx_pos_ecef_m.shape == (3, 3)
    assert len(seq.between(1.0, 3.0)) == 2  # half-open
    assert DetectionSequence().rx_pos_ecef_m.shape == (0, 3)
    with pytest.raises(ValueError):
        DetectionSequence([det(2), det(1)], sort=False)


# ---- playback ------------------------------------------------------------ #
def make_playback() -> MeasurementPlayback:
    return MeasurementPlayback.from_measurements(
        [
            det(1.0),
            KnownAircraftState(1.0, "abc123", RX),
            TxLocationHint(1.0, RX),
            det(2.5),
            KnownAircraftState(0.5, "abc123", RX),
        ]
    )


def test_playback_order_and_tie_break() -> None:
    names = [type(m).__name__ for m in make_playback()]
    assert names == [
        "KnownAircraftState", "TxLocationHint", "KnownAircraftState", "BistaticDetection",
        "BistaticDetection",
    ]  # fmt: skip


def test_playback_cursor_and_batches() -> None:
    pb = make_playback()
    assert [len(b) for b in pb.batches(1.0)] == [4, 1]  # [0.5,1.5), empty [1.5,2.5) skipped, [2.5,3.5)
    assert pb.finished
    pb.seek(2.0)
    assert pb.peek_time_s() == 2.5
    pb.reset()
    assert len(pb.advance_until(1.0)) == 1
    with pytest.raises(TypeError):
        MeasurementPlayback(detections="nope")  # type: ignore[arg-type]


# ---- buffer -------------------------------------------------------------- #
def test_buffer_online_and_file_modes() -> None:
    buf = MeasurementBuffer()
    pb = make_playback()
    assert pb.stream_to(buf) == 5
    assert buf.counts == {"detections": 2, "known_aircraft": 2, "tx_hints": 1}
    assert buf.latest_time_s == 2.5
    assert len(buf.all()) == 5
    assert len(buf.snapshot(0.0, 1.0)) == 1
    assert [len(b) for b in buf.iter_windows(1.0)] == [4, 1]
    assert buf.latest_tx_hint(0.9) is None and buf.latest_tx_hint(1.0) is not None
    assert len(buf.pop_before(1.0)) == 1 and len(buf) == 4
    with pytest.raises(TypeError):
        buf.add("junk")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        buf.add_detection(TxLocationHint(0.0, RX))  # type: ignore[arg-type]


def test_buffer_out_of_order_insert() -> None:
    buf = MeasurementBuffer()
    for t in (3.0, 1.0, 2.0):
        buf.add(det(t))
    assert list(buf.all().detections.times_s) == [1.0, 2.0, 3.0]


# ---- csv ----------------------------------------------------------------- #
DET_HDR = (
    "time_s,range_m,azimuth_deg,range_rate_mps,range_std_m,azimuth_std_deg,range_rate_std_mps,"
    "rx_x_m,rx_y_m,rx_z_m,rx_vx_mps,rx_vy_mps,rx_vz_mps"
)


def test_read_detections_csv_degrees(tmp_path: Path) -> None:
    f = tmp_path / "d.csv"
    f.write_text(f"{DET_HDR}\n# comment\n2,1000,90,-3,25,1,0.5,1,2,3,0,0,0\n1,900,0,-2,25,1,0.5,1,2,3,0,0,0\n")
    seq = read_detections_csv(f)
    assert list(seq.times_s) == [1.0, 2.0]
    assert seq[1].azimuth_rad == pytest.approx(math.pi / 2)
    assert seq[1].azimuth_std_rad == pytest.approx(math.radians(1))


def test_read_csv_errors_include_line(tmp_path: Path) -> None:
    f = tmp_path / "d.csv"
    f.write_text(f"{DET_HDR}\n1,900,0,-2,-25,1,0.5,1,2,3,0,0,0\n")
    with pytest.raises(CsvFormatError, match="line 2"):
        read_detections_csv(f)
    f.write_text("time_s,range_m\n1,2\n")
    with pytest.raises(CsvFormatError, match="missing required"):
        read_detections_csv(f)


def test_read_aircraft_geodetic_with_enu_velocity(tmp_path: Path) -> None:
    f = tmp_path / "a.csv"
    f.write_text(
        "time_s,icao,lat_deg,lon_deg,alt_m,ve_mps,vn_mps,vu_mps\n"
        "1,a1b2c3,0,0,1000,0,0,10\n"   # at lat=lon=0, 'up' is +X in ECEF
        "2,a1b2c3,0,0,1100,,,\n"
    )
    seq = read_known_aircraft_csv(f)
    assert seq[0].icao == "A1B2C3"
    np.testing.assert_allclose(seq[0].pos_ecef_m, geodetic_to_ecef(0, 0, 1000))
    np.testing.assert_allclose(seq[0].vel_ecef_mps, [10.0, 0.0, 0.0], atol=1e-9)
    assert seq[1].vel_ecef_mps is None
    assert np.isnan(seq.vel_ecef_mps[1]).all()
    assert list(seq.by_icao()) == ["A1B2C3"]


def test_read_tx_hint_csv_and_ambiguous_position(tmp_path: Path) -> None:
    f = tmp_path / "h.csv"
    f.write_text("time_s,x_m,y_m,z_m,pos_std_m\n0,1,2,3,5000\n")
    assert read_tx_hint_csv(f)[0].pos_std_m == 5000.0
    f.write_text("time_s,x_m,y_m,z_m,lat_deg,lon_deg,alt_m\n0,1,2,3,0,0,0\n")
    with pytest.raises(CsvFormatError, match="exactly one position format"):
        read_tx_hint_csv(f)


def test_playback_from_csv(tmp_path: Path) -> None:
    d, a, h = tmp_path / "d.csv", tmp_path / "a.csv", tmp_path / "h.csv"
    d.write_text(f"{DET_HDR}\n1,900,0,-2,25,1,0.5,1,2,3,0,0,0\n")
    a.write_text("time_s,icao,x_m,y_m,z_m\n0.5,abc,1,2,3\n")
    h.write_text("time_s,x_m,y_m,z_m\n0,1,2,3\n")
    pb = MeasurementPlayback.from_csv(detections_path=d, known_aircraft_path=a, tx_hint_path=h)
    assert len(pb) == 3 and pb.start_time_s == 0.0 and pb.end_time_s == 1.0
