#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAUTO_BASE="${HOME}/.webauto/data/data/annotation_dataset"
EXP_CONFIG="configs/t4/exp_t4.yaml"
DATA_CONFIG="configs/t4/dynamic/example.yaml"

usage() {
    echo "Usage: $0 [-g <gpu-id>] [-i <iteration>] [-c <ckpt-path>] [--vis-type <train|test|all>] <dataset-uuid>"
    echo ""
    echo "Options:"
    echo "  -g, --gpu <id>          CUDA device ID to use (default: 0)"
    echo "  -i, --iter <iter>       Use ckpt_it_<iter>.pth or model_it_<iter>.pth"
    echo "  -c, --ckpt <path>       Explicit checkpoint path (overrides -i)"
    echo "      --vis-type <type>   Frames to render: train, test, or all (default: all)"
    echo ""
    echo "If neither -i nor -c is given, picks the latest checkpoint found in models/."
    echo ""
    echo "Example:"
    echo "  $0 433a2328-a5a6-4790-87d3-59e930ac7020"
    echo "  $0 -i 8000 433a2328-a5a6-4790-87d3-59e930ac7020"
    echo "  $0 -g 1 -c output/.../models/ckpt_it_8000.pth 433a2328-a5a6-4790-87d3-59e930ac7020"
    exit 1
}

GPU_ID=""
ITER=""
CKPT_OVERRIDE=""
VIS_TYPE="all"

while [ $# -gt 0 ]; do
    case "$1" in
        -g|--gpu)
            [ $# -lt 2 ] && { echo "Error: $1 requires a value"; usage; }
            GPU_ID="$2"
            shift 2
            ;;
        -i|--iter)
            [ $# -lt 2 ] && { echo "Error: $1 requires a value"; usage; }
            ITER="$2"
            shift 2
            ;;
        -c|--ckpt)
            [ $# -lt 2 ] && { echo "Error: $1 requires a value"; usage; }
            CKPT_OVERRIDE="$2"
            shift 2
            ;;
        --vis-type)
            [ $# -lt 2 ] && { echo "Error: $1 requires a value"; usage; }
            VIS_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
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

[ $# -lt 1 ] && usage
UUID="$1"
shift

DATASET_DIR="${WEBAUTO_BASE}/${UUID}"
if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: Dataset not found: ${DATASET_DIR}"
    exit 1
fi

VERSION=$(ls -1 "${DATASET_DIR}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "${VERSION}" ]; then
    echo "Error: No version subdirectory found under ${DATASET_DIR}"
    exit 1
fi
SOURCE_DIR="${DATASET_DIR}/${VERSION}"

cd "${SCRIPT_DIR}"
PYTHON=".venv/bin/python"

read_yaml_value() {
    "${PYTHON}" -c "import yaml; d=yaml.safe_load(open('$1')); print(d.get('$2',''))"
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

# Resolve checkpoint
if [ -n "${CKPT_OVERRIDE}" ]; then
    CKPT="${CKPT_OVERRIDE}"
elif [ -n "${ITER}" ]; then
    CKPT="${MODELS_DIR}/ckpt_it_${ITER}.pth"
    if [ ! -f "${CKPT}" ]; then
        CKPT="${MODELS_DIR}/model_it_${ITER}.pth"
    fi
    if [ ! -f "${CKPT}" ]; then
        CKPT="${MODELS_DIR}/ckpt_it_${ITER}_good.pth"
    fi
else
    # Prefer the "_good" checkpoint; fall back to the latest model_it_*.pth or ckpt_it_*.pth
    CKPT=$(ls -1 "${MODELS_DIR}"/ckpt_it_*_good.pth 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "${CKPT}" ]; then
        CKPT=$(ls -1 "${MODELS_DIR}"/ckpt_it_*.pth 2>/dev/null | sort -V | tail -1 || true)
    fi
    if [ -z "${CKPT}" ]; then
        CKPT=$(ls -1 "${MODELS_DIR}"/model_it_*.pth 2>/dev/null | sort -V | tail -1 || true)
    fi
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
    echo "Error: checkpoint not found: ${CKPT:-<none>}" >&2
    exit 1
fi

RRD_PATH="${OUTPUT_DIR}/$(basename "${CKPT}" .pth)_$(date +%Y%m%d_%H%M%S).rrd"

echo "Dataset:    ${UUID}"
echo "Version:    ${VERSION}"
echo "Source:     ${SOURCE_DIR}"
echo "Checkpoint: ${CKPT}"
echo "UNet dir:   ${MODELS_DIR}"
echo "Output:     ${RRD_PATH}"
echo "Vis type:   ${VIS_TYPE}"
if [ -n "${GPU_ID}" ]; then
    echo "GPU:        ${GPU_ID}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
echo ""

exec "${PYTHON}" vis_rerun.py \
    -ec "${EXP_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -s "${SOURCE_DIR}" \
    -m "${CKPT}" \
    -un "${MODELS_DIR}" \
    -t "${VIS_TYPE}" \
    --save "${RRD_PATH}"
