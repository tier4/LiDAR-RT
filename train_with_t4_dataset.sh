#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAUTO_BASE="${HOME}/.webauto/data/data/annotation_dataset"

usage() {
    echo "Usage: $0 <dataset-uuid> [-- extra train.py args...]"
    echo ""
    echo "Example:"
    echo "  $0 835afe23-ff50-4883-a0b2-421e101a124b"
    echo "  $0 835afe23-ff50-4883-a0b2-421e101a124b -- -m output/model.pth"
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

SOURCE_DIR="${WEBAUTO_BASE}/${UUID}/0"

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "Error: Dataset not found: ${SOURCE_DIR}"
    exit 1
fi

echo "Dataset: ${UUID}"
echo "Source:  ${SOURCE_DIR}"
echo ""

cd "${SCRIPT_DIR}"
exec .venv/bin/python train.py \
    -ec configs/t4/exp_t4.yaml \
    -dc configs/t4/dynamic/example.yaml \
    -s "${SOURCE_DIR}" \
    "$@"
