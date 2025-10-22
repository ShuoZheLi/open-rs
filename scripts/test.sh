export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_API_KEY=190caefcc554590440e42593bfd6931f88f46f16
export WANDB_ENTITY=shuozhe


CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/grpo.py \
  --config recipes/grpo_base.yaml\
  --model_name_or_path /nfs/shuozhe/saved_model/Qwen2.5-0.5B \
  --per_device_eval_batch_size 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 3 \
  --eval_num_generations 2 \
  --entropy_coeff 0.001 \
  --dataset_name xiaodongguaAIGC/X-R1-750 \
  --mask_truncated_completions True \
  --token_entropy_percentile_threshold 0.1 \
  --output_dir data/test \