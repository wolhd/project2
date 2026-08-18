#!/usr/bin/env bash
# Regenerates the *_pb2.py bindings from proto/position.proto.
# Requires `grpcio-tools` (pip install grpcio-tools) or a system `protoc`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/src/trackerapp/proto_gen"
mkdir -p "${OUT_DIR}"
touch "${OUT_DIR}/__init__.py"

python -m grpc_tools.protoc \
  -I "${ROOT_DIR}/proto" \
  --python_out="${OUT_DIR}" \
  --pyi_out="${OUT_DIR}" \
  "${ROOT_DIR}/proto/position.proto"

echo "Generated bindings in ${OUT_DIR}"
