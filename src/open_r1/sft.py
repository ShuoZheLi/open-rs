# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from configs import SFTConfig
from utils import get_tokenizer
from utils.callbacks import get_callbacks
from utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from CustomSFTTrainer import CustomSFTTrainer


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

    # def tokenize(example):
    #     full = tokenizer(
    #         example["text"],
    #         truncation=True,
    #         padding=False,
    #         max_length=training_args.max_seq_length,
    #         return_tensors=None,
    #     )
    #     input_ids = full["input_ids"]

    #     # Mask all tokens before the assistant role begins
    #     full_text = example["text"]
    #     # assistant_start = full_text.find("Assistant:")
    #     assistant_prefix = tokenizer.apply_chat_template(
    #         [{"role": "assistant", "content": ""}],
    #         tokenize=False,
    #         add_generation_prompt=False,
    #     ).strip()
    #     assistant_start = full_text.find(assistant_prefix)

    #     if assistant_start == -1:
    #         labels = [-100] * len(input_ids)
    #     else:
    #         prefix = tokenizer(full_text[:assistant_start], add_special_tokens=False)["input_ids"]
    #         prefix_len = len(prefix)
    #         labels = [-100] * prefix_len + input_ids[prefix_len:]

    #     # if the last token is not the eos token, append it
    #     if input_ids[-1] != tokenizer.eos_token_id:
    #         full["input_ids"].append(tokenizer.eos_token_id)
    #         labels.append(-100)
    #     full["labels"] = labels
    #     return full

    ################
    # Load datasets
    ################
    def format_chat(example):
        messages = []

        if training_args.system_prompt is not None:
            messages.append({"role": "system", "content": training_args.system_prompt})
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["think_answer"]})

        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        }

    def tokenize(example):
        full = tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=training_args.max_seq_length,
            return_tensors=None,
        )
        input_ids = full["input_ids"]

        full_text = example["text"]
        marker = "<|im_start|>assistant"         # the literal prefix in your template
        idx = full_text.find(marker)

        if idx == -1:
            labels = [-100] * len(input_ids)
        else:
            # token-ize everything up to that marker
            prefix_ids = tokenizer(
                full_text[:idx],
                add_special_tokens=False
            )["input_ids"]
            # prefix_len = len(prefix_ids)
            # # mask the prompt + system + user, leave assistant’s tokens
            # labels = [-100] * prefix_len + input_ids[prefix_len:]
            marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
            prefix_len = len(prefix_ids) + len(marker_ids)
            labels = [-100] * prefix_len + input_ids[prefix_len:]


        # append EOS if missing (and mask it)
        if input_ids[-1] != tokenizer.eos_token_id:
            full["input_ids"].append(tokenizer.eos_token_id)
            labels.append(-100)

        full["labels"] = labels
        return full


    # raw_dset = load_dataset("parquet", script_args.dataset_name,)
    raw_dset = load_dataset("parquet", data_files={"train": script_args.dataset_name})
    dataset = raw_dset.map(format_chat, remove_columns=raw_dset["train"].column_names)
    dataset = dataset.map(tokenize, remove_columns=["text"])

    # decoded = tokenizer.decode(dataset["train"]['input_ids'][0])
    # visible_labels = [l if l != -100 else -1 for l in dataset["train"]['labels'][0]]
    # visible_tokens = [tokenizer.decode([t]) for t in dataset["train"]['input_ids'][0]]
    # print(list(zip(visible_tokens, visible_labels)))

    # import pdb; pdb.set_trace()

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = CustomSFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    trainer.model.config.pad_token_id = tokenizer.pad_token_id # updating model config

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
