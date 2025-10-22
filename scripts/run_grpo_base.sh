CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/zero2.yaml \
--num_processes=3 \
-- \
src/open_r1/grpo.py \
--dataset_name "/nfs/shuozhe/saved_dataset/MATH-train-MATH500-test" \
--config recipes/grpo_base.yaml \
--model_name_or_path /home/edwardhu/workspace/shuozhe/open-rs/data/Llama-3.2-1B-sft_entropy/checkpoint-42 \
--output_dir data/Llama-3.2-1B_sfted_noKL_math_7500_eval_20 \
--wandb_project "Entropy" \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--beta 0.0 \
--divergence_type "skl_approx" \
--temperature 0.7 \
--repetition_penalty 1.0 \
--entropy_coeff 0.0 \
--mask_truncated_completions True \
--num_generations 12 \
--eval_strategy "steps" \
--eval_steps 20 \
--eval_num_generations 2 \
--save_strategy "steps" \
--save_steps 200 \
--logging_steps 1   \
--logging_strategy "steps" \
--do_eval true \
--eval_on_start true \
--dataset_test_split "test" \
> ./logs/Llama-3.2-1B_sfted_noKL_math_7500_eval_20.log 2>&1
