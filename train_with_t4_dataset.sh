#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAUTO_BASE="${HOME}/.webauto/data/data/annotation_dataset"
DATA_DIR="${SCRIPT_DIR}/data/t4/dynamic/1"

usage() {
    echo "Usage: $0 <dataset-uuid> [-- extra train.py args...]"
    echo ""
    echo "Example:"
    echo "  $0 835afe23-ff50-4883-a0b2-421e101a124b"
    echo "  $0 835afe23-ff50-4883-a0b2-421e101a124b -- -ec configs/t4/exp_t4.yaml -dc configs/t4/dynamic/example.yaml"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

UUID="$1"
shift

# Skip "--" separator if present
if [ "${1:-}" = "--" ]; then
    shift
fi

SOURCE="${WEBAUTO_BASE}/${UUID}/0"

if [ ! -d "${SOURCE}" ]; then
    echo "Error: Dataset not found: ${SOURCE}"
    exit 1
fi

# Create data directory and symlinks
mkdir -p "${DATA_DIR}"

LINK_TARGETS=(annotation data input_bag map status.json)
for target in "${LINK_TARGETS[@]}"; do
    if [ ! -e "${SOURCE}/${target}" ]; then
        echo "Warning: ${SOURCE}/${target} does not exist, skipping"
        continue
    fi
    ln -sfn "${SOURCE}/${target}" "${DATA_DIR}/${target}"
done

echo "Linked dataset ${UUID} -> ${DATA_DIR}"
ls -la "${DATA_DIR}"

# Default train args
TRAIN_ARGS=(-ec configs/t4/exp_t4.yaml -dc configs/t4/dynamic/example.yaml)
if [ $# -gt 0 ]; then
    TRAIN_ARGS=("$@")
fi

echo ""
echo "Starting training..."
echo "  Command: .venv/bin/python train.py ${TRAIN_ARGS[*]}"
echo ""

cd "${SCRIPT_DIR}"
exec .venv/bin/python train.py "${TRAIN_ARGS[@]}"
