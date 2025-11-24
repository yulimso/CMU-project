#!/bin/bash

CONDA_ENV="unsloth_env"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# ## llama32-11b
# SAVE_DIR=${2:-"../results/DiffusionDB/llama32-11b"}
# # SAVE_DIR=${2:-"../results/Ours/llama32-11b"}
# MODEL_NAME=${3:-"unsloth/Llama-3.2-11B-Vision-Instruct"}

# ## llava-1.5-7b
# SAVE_DIR=${2:-"../results/DiffusionDB/llava-1.5-7b-hf"}
# # SAVE_DIR=${2:-"../results/Ours/llava-1.5-7b-hf"}
# MODEL_NAME=${3:-"unsloth/llava-1.5-7b-hf"}

## Qwen2-VL-7b
SAVE_DIR=${2:-"..results/DiffusionDB/qwen2-vl-7b"}
# SAVE_DIR=${2:-"..results/Ours/qwen2-vl-7b"}
MODEL_NAME=${3:-"unsloth/Qwen2-VL-7B-Instruct"}

# eval_metric (: bleu, rouge, cider, meteor, clip, all)
EVAL_METRIC=${4:-"all"}
SCRIPT_PATH="../eval/run.py"
# JSONL_FILE="../data/captions.jsonl"
JSONL_FILE="../data/captions_correct.jsonl"
# JSONL_FILE="../data/captions_diffdb.jsonl"

python3 $SCRIPT_PATH \
    --jsonl "$JSONL_FILE" \
    --model_name "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --device "cuda" \
    --eval_metric "$EVAL_METRIC" \
    --jsonl "$JSONL_FILE"

echo "Execution completed."