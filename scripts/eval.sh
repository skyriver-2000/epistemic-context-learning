#!/bin/bash
# Combined script: Start vLLM servers and then run MAS evaluation
# This script first deploys vLLM models, waits for them to be ready, then runs evaluation

echo "=== Starting Combined vLLM Deployment and MAS Evaluation ==="

# Export environment variables
export MAX_WORKERS_NUM=64

# ==============================
# vLLM Server Configuration
# ==============================
VLLM_MODELS=(
    # Base
    Qwen/Qwen3-4B-Instruct-2507

    # SA
    # saved/gpqa/grpo-two-stage-qwen3-4b-nonMAS-NS-TUNE-OR/checkpoint-89

    # AG
    # saved/gpqa/grpo-two-stage-qwen3-4b-AG-TUNE-OR/checkpoint-89
    # saved/gpqa/grpo-two-stage-qwen3-4b-CR-AG-TUNE-OR/checkpoint-89

    # MA-Outcome
    # saved/gpqa/grpo-two-stage-qwen3-4b-JP-TUNE-PRR/checkpoint-89
    # saved/gpqa/grpo-two-stage-qwen3-4b-NS-TUNE-OR/checkpoint-89

    # MA-Reasoning
    # saved/gpqa/grpo-two-stage-qwen3-4b-CR-JP-TUNE-PRR/checkpoint-89
    # saved/gpqa/grpo-two-stage-qwen3-4b-CR-NS-TUNE-OR/checkpoint-89
)
VLLM_IPS=(
    "0.0.0.0" 
    "0.0.0.0"
    "0.0.0.0"
    "0.0.0.0"
)
VLLM_PORT_NUMBERS=(
    "9090"     
    "9091"
    "9092"
    "9093"
)
VLLM_CUDA_DEVICES=(
    "0"
    "1"
    "2"
    "3"
)
MAX_LENGTH=40960

# ==============================
# MAS Evaluation Configuration
# ==============================
EVAL_MODELS=(
    # Base
    "Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507"

    # "deepseek/deepseek-v3.2 deepseek/deepseek-v3.2 deepseek/deepseek-v3.2 deepseek/deepseek-v3.2 deepseek/deepseek-v3.2 deepseek/deepseek-v3.2"
    # "openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini"
    # "openai/gpt-5.2 openai/gpt-5.2 openai/gpt-5.2 openai/gpt-5.2 openai/gpt-5.2 openai/gpt-5.2"
    # "google/gemini-3-flash-preview google/gemini-3-flash-preview google/gemini-3-flash-preview google/gemini-3-flash-preview google/gemini-3-flash-preview google/gemini-3-flash-preview"
    # "google/gemini-3-pro-preview google/gemini-3-pro-preview google/gemini-3-pro-preview google/gemini-3-pro-preview google/gemini-3-pro-preview google/gemini-3-pro-preview"

    # SA
    # "saved/gpqa/grpo-two-stage-qwen3-4b-nonMAS-NS-TUNE-OR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-nonMAS-NS-TUNE-OR/checkpoint-89"
    
    # AG
    # "saved/gpqa/grpo-two-stage-qwen3-4b-AG-TUNE-OR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-AG-TUNE-OR/checkpoint-89"
    # "saved/gpqa/grpo-two-stage-qwen3-4b-CR-AG-TUNE-OR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-CR-AG-TUNE-OR/checkpoint-89"
    
    # MA-Outcome
    # "saved/gpqa/grpo-two-stage-qwen3-4b-JP-TUNE-PRR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-JP-TUNE-PRR/checkpoint-89"
    # "saved/gpqa/grpo-two-stage-qwen3-4b-NS-TUNE-OR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-NS-TUNE-OR/checkpoint-89"
    
    # MA-Reasoning
    # "saved/gpqa/grpo-two-stage-qwen3-4b-CR-JP-TUNE-PRR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-CR-JP-TUNE-PRR/checkpoint-89"
    # "saved/gpqa/grpo-two-stage-qwen3-4b-CR-NS-TUNE-OR/checkpoint-89 saved/gpqa/grpo-two-stage-qwen3-4b-CR-NS-TUNE-OR/checkpoint-89"
)
EVAL_DATASETS=(
    # gpqa_formatted
    # mmlu_pro
    # gpqa_natural
    # mmlu_pro_natural
)
EVAL_CURRENT_REASON=(
    "0 0 0 1 1 1"
    # "1 1"
    # "0 0"
    # "0 1"
)
EVAL_HISTORY_REASON=(
    "0 0 0 0 0 0"
    # "0 0"
)
EVAL_REVERT_IDENTITY=(
    "0 0 0 0 0 0"
    # "1 0"
    # "2 0"
    # "0 0"
)
EVAL_IPS=(
    "0.0.0.0 0.0.0.0 0.0.0.0 0.0.0.0 0.0.0.0 0.0.0.0"
    # "0.0.0.0 0.0.0.0"
    # "0.0.0.0 0.0.0.0"
    # "0.0.0.0 0.0.0.0"
    # "0.0.0.0 0.0.0.0"
)
EVAL_PORT_NUMBERS=(
    "9090 9090 9090 9090 9090 9090"
    # "9090 9090"
    # "9091 9091"
    # "9092 9092"
    # "9093 9093"
)
EVAL_TAG_PEER=(
    # Ablation: Number of Peers
    # "2_1 2_1"
    # "3_1 3_1"
    # "2_1 2_1 2_1 3_1 3_1 3_1"

    # Ablation: Number of History Rounds
    # "4_1_2 4_1_2 4_1_2 4_1_8 4_1_8 4_1_8"
    # "4_1_2 4_1_2"
    # "4_1_8 4_1_8"

    # Default Config
    # "4_1 4_1"
    "4_1 4_1 4_1 4_1 4_1 4_1"
)
EVAL_TAG_PROMPT=(
    "AG NS JP AG NS JP"
    # "JP JP"
    # "NS NS"
    # "AG AG"
)
EVAL_DECOUPLE_BELIEF=(
    "0 0 0 0 0 0"
    # "1 1"
    # "0 0"
)
EVAL_MODE="normal"
EVAL_TAG="final"  # dont use - or _ in tag
SAVE_ROOT="ecl_eval"
LOG_FILE="${SAVE_ROOT}/EVAL_${EVAL_TAG}-${EVAL_MODE}.log"

# ==============================
# Function to check if a port is ready
# ==============================
check_port_ready() {
    local host=$1
    local port=$2
    local max_attempts=300  # 50 minutes with 10-second intervals
    local attempt=0
    
    echo "Checking if service is ready at ${host}:${port}..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s --connect-timeout 5 "http://${host}:${port}/health" > /dev/null 2>&1; then
            echo "Service at ${host}:${port} is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "Attempt ${attempt}/${max_attempts}: Service not ready yet, waiting..."
        sleep 10
    done
    
    echo "ERROR: Service at ${host}:${port} failed to become ready after $((max_attempts * 10)) seconds"
    return 1
}

# ==============================
# Function to cleanup background processes
# ==============================
cleanup() {
    echo "Cleaning up background processes..."
    for i in ${!VLLM_PIDS[@]}; do
        if kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
            echo "Terminating vLLM server with PID: ${VLLM_PIDS[$i]}"
            kill -TERM ${VLLM_PIDS[$i]} 2>/dev/null
        fi
    done
    sleep 5
    for i in ${!VLLM_PIDS[@]}; do
        if kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
            echo "Force killing vLLM server with PID: ${VLLM_PIDS[$i]}"
            kill -KILL ${VLLM_PIDS[$i]} 2>/dev/null
        fi
    done
}

# Set up signal handlers (only for interrupts, not EXIT)
trap cleanup SIGINT SIGTERM

# ==============================
# Step 1: Start vLLM Servers
# ==============================
echo "=== Step 1: Starting vLLM Servers ==="

# Start each vLLM server in background
for i in ${!VLLM_MODELS[@]}; do
    echo "Starting vLLM server for model: ${VLLM_MODELS[$i]}"
    echo "  - GPU: ${VLLM_CUDA_DEVICES[$i]}"
    echo "  - Host: ${VLLM_IPS[$i]}"
    echo "  - Port: ${VLLM_PORT_NUMBERS[$i]}"
    
    CUDA_VISIBLE_DEVICES="${VLLM_CUDA_DEVICES[$i]}" python -m vllm.entrypoints.openai.api_server \
        --model ${VLLM_MODELS[$i]} \
        --host ${VLLM_IPS[$i]} \
        --port ${VLLM_PORT_NUMBERS[$i]} \
        --tensor-parallel-size $(echo ${VLLM_CUDA_DEVICES[$i]} | tr -cd ',' | wc -c | xargs -I {} expr {} + 1) \
        --gpu-memory-utilization 0.85 \
        --max-model-len ${MAX_LENGTH} \
        --max_num_batched_tokens ${MAX_LENGTH} \
        --max-num-seqs $MAX_WORKERS_NUM \
        --dtype bfloat16 > vllm_${i}.log 2>&1 &
    
    # Store the PID for later reference
    VLLM_PIDS[$i]=$!
    echo "Started vLLM server with PID: ${VLLM_PIDS[$i]}"
done

echo "All vLLM servers started. Waiting for them to become ready..."

# ==============================
# Step 2: Wait for all vLLM servers to be ready
# ==============================
echo "=== Step 2: Waiting for vLLM Servers to be Ready ==="

all_ready=true
for i in ${!VLLM_MODELS[@]}; do
    if ! check_port_ready "${VLLM_IPS[$i]}" "${VLLM_PORT_NUMBERS[$i]}"; then
        all_ready=false
        echo "ERROR: vLLM server at ${VLLM_IPS[$i]}:${VLLM_PORT_NUMBERS[$i]} is not ready"
    fi
done

if [ "$all_ready" = false ]; then
    echo "ERROR: Not all vLLM servers are ready. Exiting..."
    exit 1
fi

echo "All vLLM servers are ready!"

# ==============================
# Step 3: Run MAS Evaluation
# ==============================
echo "=== Step 3: Starting MAS Evaluation ==="

# Create save root directory
if [ ! -d "${SAVE_ROOT}" ]; then
    mkdir -p "${SAVE_ROOT}"
fi

# Create directories for each model and run evaluation in parallel
EVAL_PIDS=()
for i in "${!EVAL_MODELS[@]}"; do
    model=${EVAL_MODELS[$i]}
    dataset=${EVAL_DATASETS[$i]}
    ip=${EVAL_IPS[$i]}
    port=${EVAL_PORT_NUMBERS[$i]}
    current_reason=${EVAL_CURRENT_REASON[$i]}
    history_reason=${EVAL_HISTORY_REASON[$i]}
    revert_identity=${EVAL_REVERT_IDENTITY[$i]}
    tag_peer=${EVAL_TAG_PEER[$i]}
    tag_prompt=${EVAL_TAG_PROMPT[$i]}
    decouple_belief=${EVAL_DECOUPLE_BELIEF[$i]}

    if [[ $dataset == *"_natural" ]]; then
        data_type="nat"
    else
        data_type="adv"
    fi

    echo "Starting evaluation for model: ${model}"
    echo "  - Data Type: ${data_type}"
    echo "  - Dataset: ${dataset}"
    echo "  - IPs: ${ip}"
    echo "  - Ports: ${port}"

    # Check if model is a space-separated list
    if [[ $model == *" "* ]]; then
        # Model is a space-separated list
        for x in $model; do
            model_dir="${SAVE_ROOT}/${dataset}/$(echo ${x} | cut -d'/' -f2- | tr '/' '_')"
            mkdir -p "${model_dir}"
        done
    else
        # Model is a single value
        model_dir="${SAVE_ROOT}/${dataset}/$(echo ${model} | cut -d'/' -f2- | tr '/' '_')"
        mkdir -p "${model_dir}"
    fi

    # Run evaluation in background
    ( 
        echo "Running MAS evaluation for ${model}..."
        python ECL/eval.py \
            --models ${model} \
            --ips ${ip} \
            --port_numbers ${port} \
            --temperature 0.7 \
            --save_root ${SAVE_ROOT}/${dataset} \
            --dataset_path final_data/${dataset}/test \
            --mode ${EVAL_MODE} \
            --tag ${EVAL_TAG} \
            --tag_peer ${tag_peer} \
            --tag_prompt ${tag_prompt} \
            --current_reason ${current_reason} \
            --history_reason ${history_reason} \
            --revert_test_identity ${revert_identity} \
            --decouple_belief ${decouple_belief} \
            --data_type ${data_type} 2>&1 | tee -a ${LOG_FILE}

        # Run analysis
        echo "Running analysis for ${model}..."
        if [[ $model == *" "* ]]; then
            # Model is a space-separated list
            for x in $model; do
                model_dir="${SAVE_ROOT}/${dataset}/$(echo ${x} | cut -d'/' -f2- | tr '/' '_')"
                python ECL/eval_analysis.py \
                    --input_dir ${model_dir} 2>&1 | tee ${model_dir}/ANALYSIS_${EVAL_TAG}-${EVAL_MODE}.log 
            done
        else
            # Model is a single value
            model_dir="${SAVE_ROOT}/${dataset}/$(echo ${model} | cut -d'/' -f2- | tr '/' '_')"
            python ECL/eval_analysis.py \
                --input_dir ${model_dir} 2>&1 | tee ${model_dir}/ANALYSIS_${EVAL_TAG}-${EVAL_MODE}.log 
        fi
        
        echo "Evaluation and analysis completed for ${model}"
    ) &
    
    # Store the PID of this evaluation process
    EVAL_PIDS+=($!)
done

echo "All evaluation processes started. Waiting for completion..."

# Wait for all evaluation processes to complete (not vLLM servers)
for pid in ${EVAL_PIDS[@]}; do
    wait $pid
done

echo "=== All evaluations completed! ==="
echo "Results can be found in: ${SAVE_ROOT}/${dataset}/"
echo "Log file: ${LOG_FILE}"

# ==============================
# Cleanup
# ==============================
cleanup

echo "=== Script completed successfully! ==="
