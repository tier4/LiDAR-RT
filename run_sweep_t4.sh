#!/bin/bash
# Helper for the configs/sweep_t4_20k.yaml hyperparameter sweep.
#
# Workflow:
#   1. Create the sweep on wandb (gets a SWEEP_ID printed):
#        ./run_sweep_t4.sh --create
#   2. On any GPU host, start an agent (or several):
#        ./run_sweep_t4.sh <SWEEP_ID>
#      Optional flags:
#        --gpu N        pin to a specific CUDA device
#        --count N      stop this agent after N runs (default: until sweep
#                       exhausts or run_cap from the yaml is hit)
#        --parallel N   launch N concurrent agent processes on the same
#                       GPU. Use this when one run only takes ~3GB so a
#                       single 24GB GPU can host 3-4 in parallel. The
#                       script forwards Ctrl+C / SIGTERM to all children.
#                       (train.py auto-suffixes the output dir with the
#                       wandb run id under WANDB_RUN_ID so parallel runs
#                       don't trample each other's checkpoints.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_CONFIG="${SCRIPT_DIR}/configs/sweep_t4_20k.yaml"
ENTITY="advanced-technology-department"
PROJECT="LiDAR-RT-debug"

usage() {
    cat <<EOF
Usage:
  $0 --create [<sweep_yaml>]                        Create a new sweep, print its SWEEP_ID
                                                    Default config: configs/sweep_t4_20k.yaml
  $0 <SWEEP_ID> [--gpu N] [--count N] [--parallel N] Run an agent for the given sweep
  $0 --resume <SWEEP_ID> [--gpu N] [--parallel N]   Alias for the run-agent form

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
    shift
    # Optional positional override: ./run_sweep_t4.sh --create configs/foo.yaml
    if [ $# -ge 1 ] && [ -f "$1" ]; then
        SWEEP_CONFIG="$1"
    elif [ $# -ge 1 ]; then
        echo "Error: config file not found: $1" >&2
        exit 1
    fi
    echo "Using sweep config: ${SWEEP_CONFIG}"
    wandb sweep \
        --project "${WANDB_PROJECT:-${PROJECT}}" \
        --entity "${WANDB_ENTITY:-${ENTITY}}" \
        "${SWEEP_CONFIG}"
    echo ""
    echo "Copy the SWEEP_ID printed above, then start an agent:"
    echo "  $0 <SWEEP_ID> [--gpu 0] [--count 10] [--parallel 3]"
    exit 0
fi

if [ "$1" = "--resume" ]; then
    shift
fi

SWEEP_ID="$1"
shift

GPU_FLAG=""
COUNT_FLAG=""
PARALLEL=1
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
        --parallel)
            [ $# -lt 2 ] && { echo "Error: --parallel requires a value"; usage; }
            PARALLEL="$2"
            if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [ "$PARALLEL" -lt 1 ]; then
                echo "Error: --parallel must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

cd "${SCRIPT_DIR}"
AGENT_URI="${WANDB_ENTITY:-${ENTITY}}/${WANDB_PROJECT:-${PROJECT}}/${SWEEP_ID}"

if [ "$PARALLEL" -eq 1 ]; then
    echo "Starting wandb agent for sweep ${SWEEP_ID} ${GPU_FLAG}"
    exec wandb agent ${COUNT_FLAG} "${AGENT_URI}"
fi

# Parallel: launch N agents in the background, forward signals.
echo "Starting ${PARALLEL} parallel wandb agents for sweep ${SWEEP_ID} ${GPU_FLAG}"
echo "  (each agent runs as its own process on the same GPU;"
echo "   train.py auto-suffixes output dirs with WANDB_RUN_ID)"

LOG_DIR="${SCRIPT_DIR}/output/sweep_agent_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

CHILD_PIDS=()
cleanup() {
    echo ""
    echo "Forwarding signal to ${#CHILD_PIDS[@]} agent(s)..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

for i in $(seq 1 "${PARALLEL}"); do
    LOG_FILE="${LOG_DIR}/agent_${TS}_${i}.log"
    echo "  agent ${i}/${PARALLEL} -> ${LOG_FILE}"
    wandb agent ${COUNT_FLAG} "${AGENT_URI}" >"${LOG_FILE}" 2>&1 &
    CHILD_PIDS+=($!)
    # Stagger startup so the OccupancyGrid cache (rebuilt on cache miss) is
    # produced by exactly one process; the others wait, find it, and load.
    if [ "$i" -lt "${PARALLEL}" ]; then
        sleep 5
    fi
done

echo "All ${PARALLEL} agents launched. PIDs: ${CHILD_PIDS[*]}"
echo "Tail any agent log with:  tail -f ${LOG_DIR}/agent_${TS}_<i>.log"
echo "Stop everything with Ctrl+C in this terminal."
wait
