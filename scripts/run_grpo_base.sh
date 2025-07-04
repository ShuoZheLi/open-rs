CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/zero2.yaml \
--num_processes=3 \
-- \
src/open_r1/grpo.py \
--dataset_name "xiaodongguaAIGC/X-R1-750" \
--config recipes/grpo_base.yaml \
--model_name_or_path /nfs/shuozhe/saved_model/Qwen2.5-1.5B \
--output_dir data/Qwen2.5-1.5B_noKL_new-kl-4e-2 \
--wandb_project "Entropy" \
--eval_strategy "no" \
--eval_steps 20 \
--logging_steps 1 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--beta 0.04 \
--entropy_coeff 0.0 \
--mask_truncated_completions True \
--save_strategy "steps" \
--save_steps 50 \
--logging_steps 1   \
--logging_strategy "steps" \
--do_eval true \
--dataset_test_split "test" \
> ./logs/Qwen2.5-1.5B_noKL_new-kl-4e-2.log 2>&1