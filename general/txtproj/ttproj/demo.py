"""End-to-end demo: write tiny CSVs, replay them, feed a buffer, process in windows.

The numbers are placeholders to exercise the I/O path, not a physical simulation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from txloc import MeasurementBuffer, MeasurementPlayback

DET_HDR = (
    "time_s,range_m,azimuth_deg,range_rate_mps,range_std_m,azimuth_std_deg,range_rate_std_mps,"
    "rx_x_m,rx_y_m,rx_z_m,rx_vx_mps,rx_vy_mps,rx_vz_mps"
)


def write_sample(folder: Path) -> tuple[Path, Path, Path]:
    det, adsb, hint = folder / "det.csv", folder / "adsb.csv", folder / "tx_hint.csv"
    det.write_text(
        DET_HDR
        + "\n"
        + "".join(
            f"{t},{20000 + 90 * t},{45 + t},-90,50,1.0,2.0,"
            f"{1500000 + 100 * t},-4500000,4200000,100,0,0\n"
            for t in range(10)
        )
    )
    adsb.write_text(
        "time_s,icao,lat_deg,lon_deg,alt_m,ve_mps,vn_mps,vu_mps\n"
        + "".join(
            f"{t},{ic},40.{t:02d},-100.0,{10000 + 10 * t},200,0,0\n"
            for t in range(0, 10, 2)
            for ic in ("A1B2C3", "D4E5F6")
        )
    )
    hint.write_text("time_s,lat_deg,lon_deg,alt_m,pos_std_m\n0,41.0,-101.0,300,20000\n")
    return det, adsb, hint


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        det, adsb, hint = write_sample(Path(tmp))
        playback = MeasurementPlayback.from_csv(
            detections_path=det, known_aircraft_path=adsb, tx_hint_path=hint
        )

    print(f"{len(playback)} events, t = {playback.start_time_s} .. {playback.end_time_s}")

    # File mode: everything in memory, then sliding windows.
    buffer = MeasurementBuffer()
    buffer.extend(playback)
    for batch in buffer.iter_windows(window_s=4.0, step_s=2.0):
        print(
            f"[{batch.t_start_s:4.1f},{batch.t_end_s:4.1f}) det={len(batch.detections)} "
            f"ac={len(batch.known_aircraft)} hint={len(batch.tx_hints)}"
        )

    # Online mode: stream events in, pop a tumbling window each time it is complete.
    playback.reset()
    live = MeasurementBuffer()
    for t_end in (5.0, 10.0):
        playback.stream_to(live, until_s=t_end)
        batch = live.pop_before(t_end)
        print(f"online batch < {t_end}: {len(batch)} items, ranges={batch.detections.ranges_m[:3]}...")


if __name__ == "__main__":
    main()
