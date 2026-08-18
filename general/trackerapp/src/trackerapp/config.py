"""Loads and validates the runtime YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class IngestConfig:
    mode: str  # "zmq" | "json_file"
    zmq_endpoint: str
    zmq_topic: str
    json_path: str
    json_poll_interval_s: float


@dataclass
class InterpolationConfig:
    step_s: float
    method: str
    max_gap_s: float


@dataclass
class AssociationConfig:
    gating_distance_m: float


@dataclass
class TrackingConfig:
    process_noise: float
    measurement_noise: float


@dataclass
class OutputConfig:
    mode: str  # "zmq" | "json_file"
    zmq_endpoint: str
    zmq_topic: str
    json_path: str


@dataclass
class AppConfig:
    ingest: IngestConfig
    interpolation: InterpolationConfig
    association: AssociationConfig
    tracking: TrackingConfig
    output: OutputConfig
    log_level: str


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())

    ingest = raw["ingest"]
    interpolation = raw["interpolation"]
    association = raw["association"]
    tracking = raw["tracking"]
    output = raw["output"]

    return AppConfig(
        ingest=IngestConfig(
            mode=ingest["mode"],
            zmq_endpoint=ingest["zmq"]["endpoint"],
            zmq_topic=ingest["zmq"].get("topic", ""),
            json_path=ingest["json_file"]["path"],
            json_poll_interval_s=float(ingest["json_file"].get("poll_interval_s", 0.0)),
        ),
        interpolation=InterpolationConfig(
            step_s=float(interpolation["step_s"]),
            method=interpolation.get("method", "linear"),
            max_gap_s=float(interpolation.get("max_gap_s", 5.0)),
        ),
        association=AssociationConfig(
            gating_distance_m=float(association.get("gating_distance_m", 500.0)),
        ),
        tracking=TrackingConfig(
            process_noise=float(tracking.get("process_noise", 0.1)),
            measurement_noise=float(tracking.get("measurement_noise", 5.0)),
        ),
        output=OutputConfig(
            mode=output["mode"],
            zmq_endpoint=output["zmq"]["endpoint"],
            zmq_topic=output["zmq"].get("topic", "tracked"),
            json_path=output["json_file"]["path"],
        ),
        log_level=raw.get("logging", {}).get("level", "INFO"),
    )
