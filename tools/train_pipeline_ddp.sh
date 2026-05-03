#!/usr/bin/env bash
set -euo pipefail

PIPELINE=${1:-configs/pipeline_512_768.yaml}
NUM_GPUS=${2:-2}

torchrun --nproc_per_node="${NUM_GPUS}" tools/train_pipeline.py --pipeline "${PIPELINE}"
