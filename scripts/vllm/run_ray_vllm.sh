#!/bin/bash

set -euo pipefail
trap 'echo "[ERROR] run_ray_vllm.sh failed at line $LINENO" >&2' ERR

########################################
# 1. Environment
########################################
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  if [ -n "${MODULES:-}" ]; then
    module load $MODULES >/dev/null 2>&1 || echo "[WARN] module load failed, continuing with current environment" >&2
  fi
fi


# Required variables (normally exported by run_benchmark.sh)
: "${GPUS_PER_NODE:?GPUS_PER_NODE is required}"
: "${CPUS_PER_NODE:?CPUS_PER_NODE is required}"
: "${NODES:?NODES is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${BENCHMARK_FILE:?BENCHMARK_FILE is required}"
: "${DATASET:?DATASET is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${LAUNCH_FOLDER:?LAUNCH_FOLDER is required}"

# Ray sanity
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_USAGE_STATS_ENABLED=1
export RAY_NUM_GPUS=$GPUS_PER_NODE
export RAY_NUM_CPUS=$CPUS_PER_NODE
export VLLM_USE_V1=0
export VLLM_USE_RAY_SPANNABLE_POOL=0
export VLLM_USE_RAY_COMPILED_DAG=0
export RAY_CGRAPH_get_timeout=1800
NB_NODES=$NODES
LOCAL_MODE="${LOCAL_EXECUTION:-false}"

########################################
# 2. Robust InfiniBand detection (IPv4 only)
########################################
IB_IFACE=$(ip -4 -o addr show | awk '{print $2}' | grep -E '^(ib|hsn|sl)' | head -n 1)

if [ -z "$IB_IFACE" ]; then
  echo "ERROR: No InfiniBand interface with IPv4 found"
  ip -4 addr show
  exit 1
fi

IB_IP_CMD="ip -4 addr show $IB_IFACE | awk '/inet / {split(\$2,a,\"/\"); print a[1]}'"

########################################
# 3. NCCL (force IB, prevent silent TCP fallback)
########################################

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=10
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=$IB_IFACE
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0

########################################
# 4. Node discovery
########################################

if [[ "$LOCAL_MODE" == "true" ]]; then
  head_node=$(hostname)
  head_node_ip="127.0.0.1"
  export RAY_HEAD_IP="$head_node_ip"
  export RAY_ADDRESS="$RAY_HEAD_IP:6379"
else
  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  nodes_array=($nodes)
  head_node=${nodes_array[0]}

  if [ "$NB_NODES" -eq 1 ]; then
      head_node_ip=$(eval "$IB_IP_CMD")
  else
      head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "$IB_IP_CMD")
  fi

  export RAY_HEAD_IP="$head_node_ip"
  export RAY_ADDRESS="$RAY_HEAD_IP:6379"
fi

########################################
# 5. Launch Ray cluster
########################################
ray stop --force || true
sleep 5

if [[ "$LOCAL_MODE" == "true" ]]; then
  ray start --head \
    --node-ip-address="$RAY_HEAD_IP" \
    --port=6379 \
    --num-cpus=$CPUS_PER_NODE \
    --num-gpus=$GPUS_PER_NODE \
    --disable-usage-stats \
    --block &
else
  echo "Slurm mode is not supported by this local launcher path. Set LOCAL_EXECUTION=true for GB10."
  exit 1
fi

########################################
# 6. Wait for Ray

########################################
sleep 60
ray status || { echo "Ray failed to start"; exit 1; }


########################################
# 7. Launch vLLM
########################################
export VLLM_RAY_USE_EXISTING_CLUSTER=1
export VLLM_HOST_IP="$RAY_HEAD_IP"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Force NCCL to use the InfiniBand interface found earlier
export NCCL_SOCKET_IFNAME=$IB_IFACE


echo "Launching vLLM on $head_node ($RAY_HEAD_IP)"
PORT=8000
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --pipeline-parallel-size $PIPELINE_PARALLEL \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host "$RAY_HEAD_IP" \
  --port $PORT \
  --trust-remote-code &
VLLM_PID=$!
########################################
# 8. Wait for Health Check & Inference Request
########################################
echo "Waiting for vLLM to initialize weights (Model: 405B)..."

# Use /v1/models instead of /health for a stricter readiness check
timeout 1800 bash -c "
until [ \"\$(curl -s -o /dev/null -w ''%{http_code}'' http://$RAY_HEAD_IP:8000/v1/models)\" == \"200\" ]; do
    echo \"Still loading weights...\"
    sleep 20
done
" || {
  echo "vLLM server failed to become ready within 30 minutes"
  kill $VLLM_PID
  exit 1
}

echo "Server is UP. Sending inference request..."

curl -X POST http://$RAY_HEAD_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Explain the concept of GPU tensor parallelism in one sentence.\"}
    ],
    \"max_tokens\": 50
  }"

echo -e "\nInference test complete."
#############################################
concurrencies=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
for conc in "${concurrencies[@]}"; do
  echo "======================================="
  echo "Running concurrency level $conc"
  echo "Results folder: $LAUNCH_FOLDER"
  echo "======================================="

  METRICS_FILE="$LAUNCH_FOLDER/gpu_metrics_${conc}.csv"
  LOG_FILE="$LAUNCH_FOLDER/logs_benchmarking_${conc}_concurrency.log"
  RESULT_FILE="$LAUNCH_FOLDER/Concurrency_${conc}.json"

  # Start GPU monitoring (all GPUs on the node where this runs)
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw,utilization.gpu,utilization.memory \
             --format=csv,noheader,nounits -l 1 > "$METRICS_FILE" &
  GPU_MON_PID=$!

  # Run benchmark against the vLLM server
  # IMPORTANT: use $RAY_HEAD_IP not localhost (unless your benchmark runs on the same head node and you know that’s true)
  set +e
  python "$BENCHMARK_FILE" \
    --backend 'vllm' \
    --host "$RAY_HEAD_IP" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --max-concurrency "$conc" \
    --num-prompts 1000 \
    --save-result \
    --result-filename "$RESULT_FILE" \
    > "$LOG_FILE" 2>&1
  RC=$?
  set -e

  # Stop monitoring
  kill "$GPU_MON_PID" >/dev/null 2>&1 || true
  sleep 2

  if [ "$RC" -ne 0 ]; then
    echo "Benchmark failed at concurrency=$conc (exit code $RC). See: $LOG_FILE"
    exit "$RC"
  fi

  echo "Done concurrency=$conc"
done

echo "All concurrency runs completed successfully."


########################################
# 9. Cleanup
########################################
echo "Shutting down..."
kill $VLLM_PID
ray stop --force
