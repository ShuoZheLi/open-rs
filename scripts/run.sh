# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_API_KEY=190caefcc554590440e42593bfd6931f88f46f16
export WANDB_ENTITY=shuozhe


CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/grpo.py \
  --config recipes/grpo.yaml \
  --cosine_max_len 3584 \
  --wandb_project Entropy\
  --dataset_name /teamspace/studios/this_studio/saved_datasets/open-rs \
  --model_name_or_path /teamspace/studios/this_studio/saved_models/DeepSeek-R1-Distill-Qwen-1.5B \
> ./logs/rs3.log 2>&1