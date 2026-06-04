#!/bin/bash
# Helper for the configs/sweep_t4_20k.yaml hyperparameter sweep.
#
# Workflow:
#   1. Create the sweep on wandb (gets a SWEEP_ID printed):
#        ./run_sweep_t4.sh --create
#   2. On any GPU host (can run multiple agents in parallel against the same
#      sweep), start an agent:
#        ./run_sweep_t4.sh <SWEEP_ID>
#      Optional: pass --gpu N to pin to a specific CUDA device.
#      Optional: pass --count N to limit the number of runs this agent does
#      before exiting (default: keep running until the sweep is exhausted or
#      run_cap from the yaml is hit).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_CONFIG="${SCRIPT_DIR}/configs/sweep_t4_20k.yaml"
ENTITY="advanced-technology-department"
PROJECT="LiDAR-RT-debug"

usage() {
    cat <<EOF
Usage:
  $0 --create                              Create a new sweep, print its SWEEP_ID
  $0 <SWEEP_ID> [--gpu N] [--count N]      Run an agent for the given sweep
  $0 --resume <SWEEP_ID> [--gpu N]         Alias for the run-agent form

Environment:
  WANDB_ENTITY   defaults to "${ENTITY}"
  WANDB_PROJECT  defaults to "${PROJECT}"
EOF
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

if [ "$1" = "--create" ]; then
    wandb sweep \
        --project "${WANDB_PROJECT:-${PROJECT}}" \
        --entity "${WANDB_ENTITY:-${ENTITY}}" \
        "${SWEEP_CONFIG}"
    echo ""
    echo "Copy the SWEEP_ID printed above, then start an agent:"
    echo "  $0 <SWEEP_ID> [--gpu 0] [--count 10]"
    exit 0
fi

if [ "$1" = "--resume" ]; then
    shift
fi

SWEEP_ID="$1"
shift

GPU_FLAG=""
COUNT_FLAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            [ $# -lt 2 ] && { echo "Error: --gpu requires a value"; usage; }
            export CUDA_VISIBLE_DEVICES="$2"
            GPU_FLAG="(GPU $2)"
            shift 2
            ;;
        --count)
            [ $# -lt 2 ] && { echo "Error: --count requires a value"; usage; }
            COUNT_FLAG="--count $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "Starting wandb agent for sweep ${SWEEP_ID} ${GPU_FLAG}"
cd "${SCRIPT_DIR}"
exec wandb agent ${COUNT_FLAG} "${WANDB_ENTITY:-${ENTITY}}/${WANDB_PROJECT:-${PROJECT}}/${SWEEP_ID}"
