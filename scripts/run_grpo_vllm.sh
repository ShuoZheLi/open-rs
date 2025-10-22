#!/bin/bash

# Set error handling
set -e

VLLM_SCRIPT="src/open_r1/vllm_serve.py"
HOST="127.0.0.1"
PORT=8000

export CUDA_VISIBLE_DEVICES=0,1

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1


python $VLLM_SCRIPT \
  --model /nfs/shuozhe/saved_model/Qwen2.5-0.5B \
  --revision main \
  --tensor_parallel_size 2 \
  --data_parallel_size 1 \
  --host $HOST \
  --port $PORT \
  --gpu_memory_utilization 0.7 \
  --dtype auto \
  --enforce_eager false \
  --kv_cache_dtype auto \
  --enable_prefix_caching true \
  --trust_remote_code true \
  --log_level info

