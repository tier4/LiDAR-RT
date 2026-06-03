#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAUTO_BASE="${HOME}/.webauto/data/data/annotation_dataset"
EXP_CONFIG="configs/t4/exp_t4.yaml"
DATA_CONFIG="configs/t4/dynamic/example.yaml"

usage() {
    echo "Usage: $0 [-g <gpu-id>] [--no-vis] [--vis-type <train|test|all>] <dataset-uuid> [-- extra train.py args...]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu <id>          CUDA device ID to use (default: 0)"
    echo "      --no-vis            Skip rerun visualization after training"
    echo "      --vis-type <type>   Frames to render: train, test, or all (default: all)"
    echo ""
    echo "Example:"
    echo "  $0 835afe23-ff50-4883-a0b2-421e101a124b"
    echo "  $0 -g 1 835afe23-ff50-4883-a0b2-421e101a124b"
    echo "  $0 -g 1 --no-vis 835afe23-ff50-4883-a0b2-421e101a124b -- -m output/model.pth"
    exit 1
}

GPU_ID=""
RUN_VIS=1
VIS_TYPE="all"

while [ $# -gt 0 ]; do
    case "$1" in
        -g|--gpu)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a value"
                usage
            fi
            GPU_ID="$2"
            shift 2
            ;;
        --no-vis)
            RUN_VIS=0
            shift
            ;;
        --vis-type)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a value"
                usage
            fi
            VIS_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            break
            ;;
        -*)
            echo "Error: Unknown option: $1"
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -lt 1 ]; then
    usage
fi

UUID="$1"
shift

# Skip "--" separator if present
if [ "${1:-}" = "--" ]; then
    shift
fi

DATASET_DIR="${WEBAUTO_BASE}/${UUID}"

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: Dataset not found: ${DATASET_DIR}"
    exit 1
fi

# Pick the highest-numbered version subdirectory (e.g. 0, 1, 2, ...)
VERSION=$(ls -1 "${DATASET_DIR}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "${VERSION}" ]; then
    echo "Error: No version subdirectory found under ${DATASET_DIR}"
    exit 1
fi
SOURCE_DIR="${DATASET_DIR}/${VERSION}"

echo "Dataset: ${UUID}"
echo "Version: ${VERSION}"
echo "Source:  ${SOURCE_DIR}"
if [ -n "${GPU_ID}" ]; then
    echo "GPU:     ${GPU_ID}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
echo ""

cd "${SCRIPT_DIR}"
PYTHON=".venv/bin/python"

"${PYTHON}" train.py \
    -ec "${EXP_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -s "${SOURCE_DIR}" \
    "$@"

if [ "${RUN_VIS}" -eq 0 ]; then
    exit 0
fi

# Resolve output dir: <model_dir>/<task_name>/<exp_name>/scene_<scene_id>/
read_yaml_value() {
    local file="$1"
    local key="$2"
    "${PYTHON}" -c "import yaml,sys; d=yaml.safe_load(open('$file')); print(d.get('$key',''))"
}

MODEL_DIR=$(read_yaml_value "${EXP_CONFIG}" model_dir)
TASK_NAME=$(read_yaml_value "${EXP_CONFIG}" task_name)
EXP_NAME=$(read_yaml_value "${EXP_CONFIG}" exp_name)
SCENE_ID=$(read_yaml_value "${DATA_CONFIG}" scene_id)
OUTPUT_DIR="${MODEL_DIR}/${TASK_NAME}/${EXP_NAME}/scene_${SCENE_ID}"
MODELS_DIR="${OUTPUT_DIR}/models"

if [ ! -d "${MODELS_DIR}" ]; then
    echo "Error: models dir not found: ${MODELS_DIR}" >&2
    exit 1
fi

# Prefer the "_good" checkpoint; fall back to the latest model_it_*.pth
CKPT=$(ls -1 "${MODELS_DIR}"/ckpt_it_*_good.pth 2>/dev/null | sort -V | tail -1 || true)
if [ -z "${CKPT}" ]; then
    CKPT=$(ls -1 "${MODELS_DIR}"/model_it_*.pth 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "${CKPT}" ]; then
    echo "Error: no checkpoint found in ${MODELS_DIR}" >&2
    exit 1
fi

RRD_PATH="${OUTPUT_DIR}/$(basename "${CKPT}" .pth)_$(date +%Y%m%d_%H%M%S).rrd"

echo ""
echo "Rendering rerun visualization"
echo "  Checkpoint: ${CKPT}"
echo "  UNet dir:   ${MODELS_DIR}"
echo "  Output:     ${RRD_PATH}"
echo ""

exec "${PYTHON}" vis_rerun.py \
    -ec "${EXP_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -s "${SOURCE_DIR}" \
    -m "${CKPT}" \
    -un "${MODELS_DIR}" \
    -t "${VIS_TYPE}" \
    --save "${RRD_PATH}"
