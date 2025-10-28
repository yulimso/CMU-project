#!/bin/bash

CONDA_ENV="unsloth_env"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

CACHE_DIR="/home/jm/cmu_project/diffusiondb_local"
EXPORT_DIR="/home/jm/cmu_project/CMU-project/data/images/diffdb"
OUT_JSONL="/home/jm/cmu_project/CMU-project/data/captions_diffdb.jsonl"
REL_PREFIX="../data/images/diffdb"

SAVE_DIR="../results/llama32-11b"
MODEL_NAME="unsloth/Llama-3.2-11B-Vision-Instruct"
SCRIPT_PATH="../eval/run.py"

python diffusiondb_loader.py \
  --cache_dir "$CACHE_DIR" \
  --subset "large_random_1k" \
  --export_dir "$EXPORT_DIR" \
  --out_jsonl "$OUT_JSONL" \
  --relative_prefix "$REL_PREFIX" \
  --start 0 --limit 1000

python3 "$SCRIPT_PATH" \
  --jsonl "$OUT_JSONL" \
  --model_name "$MODEL_NAME" \
  --save_dir "$SAVE_DIR" \
  --device "cuda" \
  --eval_metric "all"

echo "Execution completed."