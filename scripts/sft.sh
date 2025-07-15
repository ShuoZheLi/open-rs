CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/zero2.yaml \
--num_processes=4 \
-- \
src/open_r1/sft.py \
--model_name_or_path /nfs/shuozhe/saved_model/Qwen2.5-1.5B \
--dataset_name /nfs/shuozhe/data_process/cleaned_5120_sampled_with_think_answer.parquet \
--system_prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages." \
--learning_rate 2.0e-5 \
--num_train_epochs 3 \
--packing \
--max_seq_length 4096 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--bf16 \
--logging_steps 1 \
--save_strategy "epoch" \
--output_dir data/Qwen2.5-1.5B-sft_entropy_3 \
--eval_strategy "no" \
--eval_steps 100 \
--wandb_project "Entropy" \
