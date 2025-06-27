export CUDA_VISIBLE_DEVICES=3
NUM_GPUS=1
MODEL=/home/edwardhu/workspace/shuozhe/open-rs/data/R1-Distill-Qwen-1.5B-noKL/checkpoint-300
# MODEL=/nfs/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B
# MODEL=/nfs/shuozhe/saved_model/Qwen2.5-Math-1.5B-Instruct

SYSTEM_PROMPT='A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \boxed{{}}. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages.'

# This must be a valid one-line JSON string without any trailing commas
GEN_PARAMS='{"max_new_tokens":32768, "temperature":0.6, "top_p":0.95}'

MODEL_ARGS="model_name=$MODEL,\
dtype=bfloat16,\
data_parallel_size=$NUM_GPUS,\
max_model_length=32768,\
max_num_batched_tokens=32768,\
gpu_memory_utilization=0.8,\
generation_parameters=$GEN_PARAMS"


# TASK=aime24
TASK=amc23
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"

# ============================================================================================================

# export CUDA_VISIBLE_DEVICES=3
# NUM_GPUS=1

# MODEL=/home/edwardhu/workspace/shuozhe/open-rs/data/OpenRS-GRPO/checkpoint-150

# MODEL_ARGS="model_name=$MODEL,\
# dtype=bfloat16,data_parallel_size=$NUM_GPUS,\
# max_model_length=4096,\
# max_num_batched_tokens=8192,\
# gpu_memory_utilization=0.8,\
# generation_parameters={\"max_new_tokens\":512,\"temperature\":0.6,\"top_p\":0.95}"

# TASK="leaderboard|truthfulqa:mc|0|0"
# OUTPUT_DIR=data/evals/$MODEL

# lighteval vllm "$MODEL_ARGS" "$TASK" \
#   --use-chat-template \
#   --output-dir "$OUTPUT_DIR"


# export CUDA_VISIBLE_DEVICES=3
# NUM_GPUS=1

# # MODEL=/home/edwardhu/workspace/shuozhe/open-rs/data/OpenRS-GRPO/checkpoint-150
# MODEL=/nfs/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B

# MODEL_ARGS="model_name=$MODEL,\
# dtype=bfloat16,data_parallel_size=$NUM_GPUS,\
# max_model_length=2048,\
# max_num_batched_tokens=2048,\
# gpu_memory_utilization=0.7,\
# generation_parameters={\"max_new_tokens\":256,\"temperature\":0.7,\"top_p\":0.9}"

# TASK="leaderboard|truthfulqa:mc|0|0"
# OUTPUT_DIR=data/evals/${MODEL##*/}-truthfulqa-gen

# lighteval vllm "$MODEL_ARGS" "$TASK" \
#   --use-chat-template \
#   --output-dir "$OUTPUT_DIR" 


# truthfulqa

# DS 1.5B distill
# |           Task            |Version|    Metric    |Value |   |Stderr|
# |---------------------------|------:|--------------|-----:|---|-----:|
# |all                        |       |truthfulqa_mc1|0.3452|±  |0.0166|
# |                           |       |truthfulqa_mc2|0.5017|±  |0.0153|
# |leaderboard:truthfulqa:mc:0|      0|truthfulqa_mc1|0.3452|±  |0.0166|
# |                           |       |truthfulqa_mc2|0.5017|±  |0.0153|

# 150 ckps
# |           Task            |Version|    Metric    |Value |   |Stderr|
# |---------------------------|------:|--------------|-----:|---|-----:|
# |all                        |       |truthfulqa_mc1|0.3488|±  |0.0167|
# |                           |       |truthfulqa_mc2|0.5013|±  |0.0154|
# |leaderboard:truthfulqa:mc:0|      0|truthfulqa_mc1|0.3488|±  |0.0167|
# |                           |       |truthfulqa_mc2|0.5013|±  |0.0154|


# amc23

# DS
# |     Task     |Version|     Metric     |Value|   |Stderr|
# |--------------|------:|----------------|----:|---|-----:|
# |all           |       |extractive_match| 0.75|±  |0.0693|
# |custom:amc23:0|      1|extractive_match| 0.75|±  |0.0693|

# 50 ckps
# |     Task     |Version|     Metric     |Value|   |Stderr|
# |--------------|------:|----------------|----:|---|-----:|
# |all           |       |extractive_match| 0.75|±  |0.0693|
# |custom:amc23:0|      1|extractive_match| 0.75|±  |0.0693|

# 100 ckps
# |     Task     |Version|     Metric     |Value|   |Stderr|
# |--------------|------:|----------------|----:|---|-----:|
# |all           |       |extractive_match|  0.7|±  |0.0734|
# |custom:amc23:0|      1|extractive_match|  0.7|±  |0.0734|

# 150 ckps
# |     Task     |Version|     Metric     |Value|   |Stderr|
# |--------------|------:|----------------|----:|---|-----:|
# |all           |       |extractive_match|0.725|±  |0.0715|
# |custom:amc23:0|      1|extractive_match|0.725|±  |0.0715|


# aime24

# Qwen2.5-Math-1.5B-Instruct
# |     Task      |Version|     Metric     |Value|   |Stderr|
# |---------------|------:|----------------|----:|---|-----:|
# |all            |       |extractive_match|  0.1|±  |0.0557|
# |custom:aime24:0|      1|extractive_match|  0.1|±  |0.0557|


# DS 1.5B distill
# |     Task      |Version|     Metric     |Value |   |Stderr|
# |---------------|------:|----------------|-----:|---|-----:|
# |all            |       |extractive_match|0.3667|±  |0.0895|
# |custom:aime24:0|      1|extractive_match|0.3667|±  |0.0895|


# 50 ckps
# |     Task      |Version|     Metric     |Value |   |Stderr|
# |---------------|------:|----------------|-----:|---|-----:|
# |all            |       |extractive_match|0.2667|±  |0.0821|
# |custom:aime24:0|      1|extractive_match|0.2667|±  |0.0821|


# 100 ckps
#   |     Task      |Version|     Metric     |Value |   |Stderr|
# |---------------|------:|----------------|-----:|---|-----:|
# |all            |       |extractive_match|0.3667|±  |0.0895|
# |custom:aime24:0|      1|extractive_match|0.3667|±  |0.0895|

# 150 ckps
# |     Task      |Version|     Metric     |Value|   |Stderr|
# |---------------|------:|----------------|----:|---|-----:|
# |all            |       |extractive_match|  0.3|±  |0.0851|
# |custom:aime24:0|      1|extractive_match|  0.3|±  |0.0851|