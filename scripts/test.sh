export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_API_KEY=190caefcc554590440e42593bfd6931f88f46f16
export WANDB_ENTITY=shuozhe


CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=1 \
  src/open_r1/grpo.py \
  --config recipes/grpo.yaml\
  --model_name_or_path /data/shuozhe/saved_model/Qwen2.5-0.5B \
  --per_device_eval_batch_size 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_generations 2 \
  --dataset_name /data/shuozhe/saved_dataset/open-rs \