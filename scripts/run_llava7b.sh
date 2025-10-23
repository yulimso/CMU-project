#!/bin/bash

CONDA_ENV="unsloth_env"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

SAVE_DIR=${2:-"../results/llava-1.5-7b-hf"}
MODEL_NAME=${3:-"unsloth/llava-1.5-7b-hf"}
# eval_metric (: bleu, rouge, cider, meteor, clip, all)
EVAL_METRIC=${4:-"all"}
SCRIPT_PATH="eval/run_llava7b.py"
JSONL_FILE="data/captions.jsonl"

python3 $SCRIPT_PATH \
    --jsonl "$JSONL_FILE" \
    --model_name "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --device "cuda" \
    --eval_metric "$EVAL_METRIC" \
    --jsonl "$JSONL_FILE"

echo "Execution completed."