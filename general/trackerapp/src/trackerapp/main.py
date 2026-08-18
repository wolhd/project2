"""CLI entrypoint: `python -m trackerapp.main --config path/to/config.yaml`."""

from __future__ import annotations

import argparse
import logging

from trackerapp.config import load_config
from trackerapp.pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="trackerapp position pipeline")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="path to the YAML config file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
