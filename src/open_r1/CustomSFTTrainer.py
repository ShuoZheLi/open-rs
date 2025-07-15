from trl import SFTTrainer
from transformers import Trainer
import torch
import re
import numpy as np
import wandb
import pandas as pd
import torch.nn.functional as F


class CustomSFTTrainer(SFTTrainer):
    def format_reward(self, completions):
        """
        Reward if the string contains </think> followed by optional whitespace/newlines and a well-formed <answer>...</answer>.
        """
        pattern = r"</think>\s*<answer>.*?</answer>"
        matches = [re.search(pattern, content, flags=re.DOTALL) for content in completions]
        rewards = [1.0 if match else 0.0 for match in matches]
        return np.array(rewards)
    
    def log_custom(self, prompts_to_log, completions_to_log, rewards):
        if self.accelerator.is_main_process:
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:

                # For logging
                table = {
                    "step": [str(self.state.global_step)] * len(rewards),
                    "prompt": prompts_to_log,
                    "completion": completions_to_log,
                    "reward": rewards,
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def entropy_from_logits(self, logits, chunk_size: int = 1) -> torch.Tensor:
        """
        Compute the Shannon entropy (in nats) for each row of *logits* without
        materialising the full soft-max in memory.
        The batch dimension is processed in chunks of size `chunk_size` so that
        only a subset of rows is expanded to probabilities at any one time.
        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
                leading dimensions are preserved.
            chunk_size (`int`, *optional*, defaults to `1`):
                Number of rows to process per iteration.
        Returns:
            `torch.Tensor`:
                Entropy values with shape `logits.shape[:-1]`.
        """
        with torch.no_grad():
            per_token_entropies = []
            for logits_chunk in logits.split(chunk_size, dim=0):
                logps = F.log_softmax(logits_chunk, dim=-1)
                chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
                per_token_entropies.extend(chunk_entropy)

            per_token_entropies = torch.stack(per_token_entropies)
        return per_token_entropies

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(torch.tensor(inputs["position_ids"].size(1))).sum().item()
                )
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # custom logging start
            entropy_from_logits = self.entropy_from_logits(shift_logits)
            entropy_from_logits = self.accelerator.gather_for_metrics(entropy_from_logits.mean().unsqueeze(0))
            entropy_mean = entropy_from_logits.mean().item()
            self._metrics[mode]["response_entropy_avg"].append(entropy_mean)

            decode_inputs = self.processing_class.batch_decode(
                inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            decode_predictions = self.processing_class.batch_decode(predictions, skip_special_tokens=True)
            decode_inputs = self.accelerator.gather_for_metrics(decode_inputs)
            decode_predictions = self.accelerator.gather_for_metrics(decode_predictions)
            if self.accelerator.is_main_process:
                # for each response in decode_predictions, 
                # take everthing before the first </answer> tag, including </answer>
                rewards = self.format_reward([
                    re.split(r"</answer>", response, maxsplit=1)[0] + "</answer>" for response in decode_predictions
                ])
                format_accuracy = np.mean(rewards)
                self.log_custom(decode_inputs, decode_predictions, rewards)
                self._metrics[mode]["format_accuracy"].append(format_accuracy)
            # log prediction length
            prediction_lengths = [len(pred.split()) for pred in decode_predictions]
            prediction_lengths = self.accelerator.gather_for_metrics(prediction_lengths)
            if self.accelerator.is_main_process:
                self._metrics[mode]["prediction_length"].append(np.mean(prediction_lengths))
            # custom logging end

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss
    