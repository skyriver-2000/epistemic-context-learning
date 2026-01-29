#!/bin/bash
RECIPE_ROOT="recipes/train_configs"

CONFIGS=(
    grpo_two_stage-qwen3-4b-MAS-CR-DB-NS-OR-mmlupro.yaml
    grpo_two_stage-qwen3-4b-MAS-DB-NS-OR-mmlupro.yaml
)

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

for dir in saved logs; do
  [ -d "$dir" ] || mkdir "$dir"
done

for config in "${CONFIGS[@]}"; do
    training=$(basename "$config" .yaml | cut -d'-' -f1)
    task=$(basename "$config" .yaml)
    
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    MODEL_NAME=$(awk -F': ' '/model_name_or_path:/ {print $2; exit}' "$RECIPE_ROOT/$config" | awk '{print $1}')

    echo "Starting training for $task"

    ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file "recipes/accelerate_configs/ds2.yaml" \
    --num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
    --main_process_port 29512 \
    ECL/train.py "$RECIPE_ROOT/$config" 2>&1 | tee "logs/${task}.log"

    sleep 5
done