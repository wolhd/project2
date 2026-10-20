"""Complete execution path:  CSV files -> playback -> buffer -> association + least squares -> tx estimate.

A synthetic scenario provides the CSVs and the truth so the result can be scored.
Usage:  python examples/run_estimate.py [--seed N] [--keep DIR]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from txloc import MeasurementBuffer, MeasurementPlayback, StaticTxEstimator
from txloc.geodesy import ecef_to_geodetic
from txloc.simulate import CLUTTER, UNKNOWN, ScenarioConfig, make_scenario, write_csvs


def run(folder: Path, seed: int) -> None:
    scenario = make_scenario(ScenarioConfig(seed=seed))
    det_csv, adsb_csv, hint_csv = write_csvs(scenario, folder)

    # 1) read the three CSVs and replay them in time order
    playback = MeasurementPlayback.from_csv(
        detections_path=det_csv, known_aircraft_path=adsb_csv, tx_hint_path=hint_csv
    )
    print(f"playback: {len(playback)} events, t = {playback.start_time_s} .. {playback.end_time_s} s")

    # 2) file mode: put everything in the buffer, take it as one batch
    buffer = MeasurementBuffer()
    buffer.extend(playback)
    batch = buffer.all()
    print(f"batch: {len(batch.detections)} detections, {len(batch.known_aircraft)} ADS-B reports, "
          f"{len(batch.tx_hints)} hint(s); counts by truth label: "
          f"{dict(Counter(v if v in (CLUTTER, UNKNOWN) else 'KNOWN' for v in scenario.labels.values()))}")

    # 3) estimate
    t0 = time.perf_counter()
    est = StaticTxEstimator().estimate(batch)
    elapsed = time.perf_counter() - t0

    # 4) report and score against truth
    truth = scenario.tx_truth.pos_ecef_m
    err = est.position_ecef_m - truth
    nees = float(err @ np.linalg.solve(est.covariance_m2, err))
    lat, lon, alt = ecef_to_geodetic(est.position_ecef_m)
    hint_err = float(np.linalg.norm(scenario.hint.pos_ecef_m - truth))

    correct = sum(
        1 for a in est.assignments
        if scenario.labels[(batch.detections[a.detection_index].time_s,
                            batch.detections[a.detection_index].range_m)] == a.icao
    )  # fmt: skip
    print(f"\nconverged={est.converged} rounds={est.association_rounds} time={elapsed:.1f}s  ({est.message})")
    print(f"estimate: lat {lat:.5f}  lon {lon:.5f}  alt {alt:.0f} m")
    print(f"1-sigma (ECEF x,y,z): {np.round(est.position_std_m, 1)} m")
    print(f"error: {np.linalg.norm(err):.1f} m   (hint was {hint_err:.0f} m off)   NEES={nees:.2f} (3 dof, 99% < 11.3)")
    print(f"associations: {len(est.assignments)} accepted, {correct} correct "
          f"({correct / max(len(est.assignments), 1):.0%} precision)")
    print(f"principal 1-sigma axes (1 / singular values): {np.round(1.0 / est.singular_values, 1)} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep", type=Path, help="write the CSVs here instead of a temp dir")
    args = parser.parse_args()
    if args.keep:
        run(args.keep, args.seed)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp), args.seed)


if __name__ == "__main__":
    main()
