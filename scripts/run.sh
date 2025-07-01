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
  --beta 0 \
  --entropy_coeff 0.0 \
  --mask_truncated_completions True \
  --num_train_epochs 2\
  --wandb_project Entropy\
  --output_dir data/R1-Distill-Qwen-1.5B-noKL-1500-mask_truncated \
  --dataset_name xiaodongguaAIGC/X-R1-1500 \
  --model_name_or_path /nfs/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B \
> ./logs/R1-Distill-Qwen-1.5B-noKL-1500-mask_truncated.log 2>&1


  # --beta 0.0001 \