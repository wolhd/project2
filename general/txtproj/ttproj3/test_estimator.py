from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from txloc import (
    BistaticMeasurementModel,
    EstimatorConfig,
    KnownAircraftSequence,
    KnownAircraftState,
    KnownAircraftTrack,
    MeasurementBatch,
    MeasurementBuffer,
    MeasurementComponent as C,
    MeasurementPlayback,
    StaticTxEstimator,
    TxEstimate,
    TxHintSequence,
)
from txloc.simulate import Scenario, ScenarioConfig, make_scenario, write_csvs

FAST = EstimatorConfig(n_hypotheses=40, n_scoring_detections=100)


def small(seed: int) -> ScenarioConfig:
    return ScenarioConfig(seed=seed, duration_s=60.0, n_unknown_aircraft=6)


def run(scenario: Scenario) -> tuple[MeasurementBatch, TxEstimate]:
    batch = MeasurementBatch(
        detections=scenario.detections, known_aircraft=scenario.adsb, tx_hints=TxHintSequence([scenario.hint])
    )
    return batch, StaticTxEstimator(FAST).estimate(batch)


@pytest.mark.parametrize("seed", [7, 1])
def test_recovers_transmitter_from_cluttered_detections(seed: int) -> None:
    sc = make_scenario(small(seed))
    batch, est = run(sc)
    assert est.converged
    err = est.position_ecef_m - sc.tx_truth.pos_ecef_m
    hint_err = np.linalg.norm(sc.hint.pos_ecef_m - sc.tx_truth.pos_ecef_m)
    assert np.linalg.norm(err) < 0.1 * hint_err
    nees = float(err @ np.linalg.solve(est.covariance_m2, err))
    assert nees < 16.3  # 99.9 % chi-square, 3 dof: the reported covariance is honest

    labels = [
        sc.labels[(batch.detections[a.detection_index].time_s, batch.detections[a.detection_index].range_m)]
        for a in est.assignments
    ]
    wrong = sum(1 for lab, a in zip(labels, est.assignments) if lab != a.icao)
    assert wrong <= 0.03 * len(labels)  # near-perfect precision


def test_csv_round_trip_feeds_the_same_estimate(tmp_path: Path) -> None:
    sc = make_scenario(small(3))
    det, adsb, hint = write_csvs(sc, tmp_path)
    playback = MeasurementPlayback.from_csv(detections_path=det, known_aircraft_path=adsb, tx_hint_path=hint)
    buf = MeasurementBuffer()
    buf.extend(playback)
    batch = buf.all()
    assert len(batch.detections) == len(sc.detections)
    np.testing.assert_array_equal(batch.detections.ranges_m, sc.detections.ranges_m)  # repr round trip is exact
    direct = StaticTxEstimator(FAST).estimate(
        MeasurementBatch(detections=sc.detections, known_aircraft=sc.adsb, tx_hints=TxHintSequence([sc.hint]))
    )
    via_csv = StaticTxEstimator(FAST).estimate(batch)
    np.testing.assert_allclose(via_csv.position_ecef_m, direct.position_ecef_m, atol=1e-3)


def test_estimator_edge_cases() -> None:
    sc = make_scenario(small(3))
    est = StaticTxEstimator(FAST)
    with pytest.raises(ValueError, match="hint"):
        est.estimate(MeasurementBatch(detections=sc.detections, known_aircraft=sc.adsb))
    with pytest.raises(TypeError):
        est.estimate("batch")  # type: ignore[arg-type]
    no_aircraft = est.estimate(MeasurementBatch(detections=sc.detections, tx_hints=TxHintSequence([sc.hint])))
    assert not no_aircraft.converged
    np.testing.assert_allclose(no_aircraft.position_ecef_m, sc.hint.pos_ecef_m)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        EstimatorConfig(loss="l2")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EstimatorConfig(n_hypotheses=0)
    with pytest.raises(ValueError):
        EstimatorConfig(extra_range_std_m=-1.0)


def test_extra_std_is_added_in_quadrature() -> None:
    sc = make_scenario(small(3))
    det = sc.detections[0]
    plain = BistaticMeasurementModel().noise_covariance(det)
    extra = BistaticMeasurementModel(extra_std={C.RANGE: 30.0}).noise_covariance(det)
    assert extra[0, 0] == pytest.approx(plain[0, 0] + 900.0)
    assert extra[1, 1] == plain[1, 1]
    with pytest.raises(ValueError):
        BistaticMeasurementModel(extra_std={C.RANGE: -1.0})


def test_track_velocity_comes_from_reported_velocity_not_noisy_positions() -> None:
    v = np.array([100.0, 0.0, 0.0])
    seq = KnownAircraftSequence(
        [
            KnownAircraftState(0.0, "X", [0.0, 0.0, 0.0], v),
            KnownAircraftState(4.0, "X", [400.0, 50.0, 0.0], v + [0.0, 0.2, 0.0]),  # +50 m position error
        ]
    )
    s = KnownAircraftTrack(seq).state_at(2.0)
    assert s is not None
    np.testing.assert_allclose(s.vel_ecef_mps, [100.0, 0.1, 0.0])  # blend of reports, not a 12 m/s derivative
