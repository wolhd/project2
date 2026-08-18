# trackerapp

A modular position-processing pipeline:

```
ingest (zmq | json file, protobuf-decoded)
    -> geodetic -> ECEF coordinate transform
    -> interpolation (fill gaps / resample in time)
    -> association          (external/future library)
    -> state tracking       (external/future library)
    -> output wrapper (zmq | json file)
```

## Layout

```
trackerapp/
├── config/
│   └── default.yaml          # runtime configuration (ingest/output mode, params)
├── proto/
│   └── position.proto        # wire format: lat/lon/alt/time message
├── src/trackerapp/
│   ├── main.py                # entrypoint / CLI
│   ├── config.py              # config loading & validation
│   ├── pipeline.py            # wires the stages together, owns the run loop
│   ├── models/
│   │   └── types.py           # internal dataclasses shared across stages
│   ├── ingest/
│   │   ├── base.py            # SourceReader interface
│   │   ├── zmq_reader.py       # protobuf-over-zmq subscriber
│   │   └── json_file_reader.py # json file / line-delimited json reader
│   ├── transforms/
│   │   └── geodetic_ecef.py    # WGS84 geodetic <-> ECEF conversion
│   ├── interpolation/
│   │   └── interpolator.py     # time-based interpolation of position tracks
│   ├── association/
│   │   └── base.py             # Associator interface (future library plugs in here)
│   ├── tracking/
│   │   └── base.py             # StateTracker interface (future library plugs in here)
│   └── output/
│       ├── base.py             # ResultPublisher interface
│       ├── zmq_publisher.py    # protobuf-over-zmq publisher
│       └── json_file_writer.py # json file writer
├── tests/                      # unit tests, one module per stage
├── scripts/
│   ├── generate_proto.sh       # regenerates *_pb2.py from proto/position.proto
│   └── run.sh                  # convenience launcher
└── pyproject.toml
```

## Design notes

- **Interfaces first.** `ingest/base.py`, `association/base.py`, `tracking/base.py`,
  and `output/base.py` define abstract base classes. Concrete zmq/json
  implementations exist now; association and tracking only have a
  `PassThrough*` stub implementation until the real libraries are available.
  Swapping in the future libraries means writing one adapter class per stage,
  not touching `pipeline.py`.
- **Internal representation is ECEF.** Ingest stages decode raw messages into
  `models.types.GeodeticPosition`, then `transforms.geodetic_ecef` converts to
  `models.types.EcefPosition` before anything else touches the data. Every
  downstream stage (interpolation, association, tracking) speaks ECEF only.
- **Output converts back.** The output wrapper is responsible for turning the
  tracker's ECEF result back into whatever the sink expects (currently
  geodetic lat/lon/alt in the outgoing message, see `output/base.py`).
- **Config-driven mode selection.** `config/default.yaml` picks zmq vs. json
  file for both ingest and output so you don't need code changes to switch
  between live and offline runs.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
bash scripts/generate_proto.sh      # produces src/trackerapp/proto_gen/position_pb2.py
python -m trackerapp.main --config config/default.yaml
```
