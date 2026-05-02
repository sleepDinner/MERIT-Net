#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default_512.yaml}
NUM_GPUS=${2:-2}

torchrun --nproc_per_node="${NUM_GPUS}" tools/train.py --config "${CONFIG}"
