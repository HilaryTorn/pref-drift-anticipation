"""TRL PPO trainer with episode-level masking for capped completions.

TRL 1.9's experimental PPO loop treats a response that used every available
generation token as an ordinary scored episode.  This subclass keeps the
upstream loop structure but makes termination a hard eligibility condition:
unfinished episodes contribute to no RM objective statistic, whitening,
advantage, policy loss, value loss, or optimizer update.
"""

from __future__ import annotations

import gc
import math
import time

import numpy as np
import torch
from transformers import GenerationConfig
from trl.experimental.ppo import PPOTrainer
import trl.experimental.ppo.ppo_trainer as ppo_impl

from scripts.rl_training.completion_semantics import (
    normalize_eos_token_ids,
    tensor_completion_termination_mask,
    truncate_responses_at_eos,
)


def selective_log_softmax_token_chunks(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Gather token log-probs without materializing a full-sequence softmax.

    TRL's bfloat16 implementation chunks only on the batch dimension.  PPO's
    micro-batch is one sequence, so a long completion still materializes a
    ``response_length x vocab_size`` softmax and can exceed colocated GPU
    memory.  Chunking the token dimension is mathematically identical while
    bounding that temporary allocation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}"
        )
    return torch.cat(
        [
            ppo_impl.selective_log_softmax(logit_chunk, label_chunk)
            for logit_chunk, label_chunk in zip(
                logits.split(chunk_size, dim=-2),
                labels.split(chunk_size, dim=-1),
                strict=True,
            )
        ],
        dim=-1,
    )


def entropy_from_logits_token_chunks(
    logits: torch.Tensor, *, chunk_size: int = 128
) -> torch.Tensor:
    """Compute per-token entropy with bounded full-vocabulary temporaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = []
    for logit_chunk in logits.split(chunk_size, dim=-2):
        log_probs = torch.nn.functional.log_softmax(logit_chunk, dim=-1)
        chunks.append(-(log_probs.exp() * log_probs).sum(dim=-1))
    return torch.cat(chunks, dim=-1)


def masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean whose all-masked result is a differentiable zero."""
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def masked_whiten_or_zero(
    values: torch.Tensor, mask: torch.Tensor, *, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten eligible values, returning zero when variance is undefined."""
    mask_count = int(mask.sum().item())
    if mask_count <= 1:
        return torch.zeros_like(values)
    return ppo_impl.masked_whiten(values, mask=mask, shift_mean=shift_mean)


class TruncationMaskedPPOTrainer(PPOTrainer):
    """Experimental PPO with strict completed-episode-only training semantics."""

    def __init__(self, *args, eos_token_ids, **kwargs):
        self.termination_eos_token_ids = normalize_eos_token_ids(eos_token_ids)
        if not self.termination_eos_token_ids:
            raise ValueError("PPO requires at least one authoritative EOS token id")
        self.completion_masking_stats = {
            "scheduled_episodes": 0,
            "terminated_episodes": 0,
            "truncated_episodes": 0,
            "effective_updates": 0,
            "all_truncated_batches": 0,
        }
        super().__init__(*args, **kwargs)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": args.temperature + 1e-7,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "eos_token_id": list(self.termination_eos_token_ids),
            "pad_token_id": processing_class.pad_token_id,
        }
        generation_config = GenerationConfig(**generation_kwargs)

        accelerator.print(
            "===training policy (completed episodes only; EOS ids "
            f"{list(self.termination_eos_token_ids)})==="
        )
        start_time = time.time()
        stats_shape = (
            args.num_ppo_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        model.train()

        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        for name in ("logging_steps", "eval_steps", "save_steps"):
            value = getattr(args, name)
            if value is not None:
                setattr(
                    self.state,
                    name,
                    math.ceil(self.state.max_steps * value) if value < 1 else value,
                )
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            approxkl_stats = torch.zeros(stats_shape, device=device)
            pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
            pg_loss_stats = torch.zeros(stats_shape, device=device)
            vf_loss_stats = torch.zeros(stats_shape, device=device)
            vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
            entropy_stats = torch.zeros(stats_shape, device=device)
            ratio_stats = torch.zeros(stats_shape, device=device)
            valid_stat_mask = torch.zeros(stats_shape, dtype=torch.bool, device=device)

            self.state.episode += args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses_parts = []
                postprocessed_parts = []
                logprobs_parts = []
                ref_logprobs_parts = []
                scores_parts = []
                sequence_lengths_parts = []
                values_parts = []
                with ppo_impl.unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=generation_kwargs,
                ) as unwrapped_model:
                    query_responses, logitss = ppo_impl.batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax_token_chunks(logits, response)
                    del logits
                    ppo_impl.empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = ppo_impl.forward(
                                model.policy, query_response, processing_class.pad_token_id
                            )
                    else:
                        ref_output = ppo_impl.forward(
                            ref_policy, query_response, processing_class.pad_token_id
                        )
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax_token_chunks(
                        ref_logits, response
                    )
                    del ref_output, ref_logits
                    ppo_impl.empty_cache()

                    postprocessed_response = truncate_responses_at_eos(
                        response,
                        self.termination_eos_token_ids,
                        processing_class.pad_token_id,
                    )
                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), dim=1
                    )
                    sequence_length = (
                        ppo_impl.first_true_indices(
                            postprocessed_response == processing_class.pad_token_id
                        )
                        - 1
                    )
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = ppo_impl.get_reward(
                        unwrapped_value_model,
                        query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = ppo_impl.get_reward(
                        reward_model,
                        postprocessed_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )

                    responses_parts.append(response)
                    postprocessed_parts.append(postprocessed_response)
                    logprobs_parts.append(logprob)
                    ref_logprobs_parts.append(ref_logprob)
                    sequence_lengths_parts.append(sequence_length)
                    scores_parts.append(score)
                    values_parts.append(value)

                responses = torch.cat(responses_parts, 0)
                postprocessed_responses = torch.cat(postprocessed_parts, 0)
                logprobs = torch.cat(logprobs_parts, 0)
                ref_logprobs = torch.cat(ref_logprobs_parts, 0)
                sequence_lengths = torch.cat(sequence_lengths_parts, 0)
                scores = torch.cat(scores_parts, 0)
                values = torch.cat(values_parts, 0)
                del logprob, ref_logprob, full_value, value, score, unwrapped_model
                ppo_impl.empty_cache()
                gc.collect()

                terminated = tensor_completion_termination_mask(
                    responses, self.termination_eos_token_ids
                )
                gathered_terminated = accelerator.gather_for_metrics(terminated.to(torch.long))
                n_scheduled = int(gathered_terminated.numel())
                n_terminated = int(gathered_terminated.sum().item())
                n_truncated = n_scheduled - n_terminated
                globally_has_valid = n_terminated > 0
                self.completion_masking_stats["scheduled_episodes"] += n_scheduled
                self.completion_masking_stats["terminated_episodes"] += n_terminated
                self.completion_masking_stats["truncated_episodes"] += n_truncated
                if globally_has_valid:
                    self.completion_masking_stats["effective_updates"] += 1
                else:
                    self.completion_masking_stats["all_truncated_batches"] += 1

                response_idxs = torch.arange(
                    responses.shape[1], device=responses.device
                ).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > sequence_lengths_p1.unsqueeze(1)
                # The decisive correction: an unfinished episode has no valid
                # policy or value tokens, regardless of how much text it emitted.
                padding_mask |= ~terminated.unsqueeze(1)
                padding_mask_p1 |= ~terminated.unsqueeze(1)
                logprobs = torch.masked_fill(
                    logprobs, padding_mask, ppo_impl.INVALID_LOGPROB
                )
                ref_logprobs = torch.masked_fill(
                    ref_logprobs, padding_mask, ppo_impl.INVALID_LOGPROB
                )
                values = torch.masked_fill(values, padding_mask_p1, 0)
                scores = torch.masked_fill(scores, ~terminated, 0)

                logr = ref_logprobs - logprobs
                kl = (
                    -logr
                    if args.kl_estimator == "k1"
                    else (logr.exp() - 1) - logr
                )
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(
                    sequence_lengths_p1 < rewards.size(1),
                    sequence_lengths_p1,
                    sequence_lengths,
                )
                rewards[actual_start, actual_end] += scores
                rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                if args.whiten_rewards:
                    rewards = masked_whiten_or_zero(
                        rewards, ~padding_mask_p1, shift_mean=False
                    )
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                lastgaelam = 0
                advantages_reversed = []
                for t in reversed(range(responses.shape[1])):
                    nextvalues = values[:, t + 1] if t < responses.shape[1] - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], dim=1)
                returns = advantages + values
                advantages = masked_whiten_or_zero(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                returns = torch.masked_fill(returns, padding_mask_p1, 0)
                ppo_impl.empty_cache()

            if globally_has_valid:
                for ppo_epoch_idx in range(args.num_ppo_epochs):
                    b_inds = np.random.permutation(args.local_batch_size)
                    minibatch_idx = 0
                    accumulated_has_valid = False
                    for mini_batch_start in range(
                        0, args.local_batch_size, args.local_mini_batch_size
                    ):
                        mini_batch_end = mini_batch_start + args.local_mini_batch_size
                        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                        gradient_accumulation_idx = 0
                        for micro_batch_start in range(
                            0,
                            args.local_mini_batch_size,
                            args.per_device_train_batch_size,
                        ):
                            with accelerator.accumulate(model):
                                micro_batch_end = (
                                    micro_batch_start + args.per_device_train_batch_size
                                )
                                micro_batch_inds = mini_batch_inds[
                                    micro_batch_start:micro_batch_end
                                ]
                                mb_advantage = advantages[micro_batch_inds]
                                mb_responses = responses[micro_batch_inds]
                                mb_query_responses = query_responses[micro_batch_inds]
                                mb_logprobs = logprobs[micro_batch_inds]
                                mb_return = returns[micro_batch_inds]
                                mb_values = values[micro_batch_inds]
                                policy_mask = ~padding_mask[micro_batch_inds]
                                value_mask = ~padding_mask_p1[micro_batch_inds]
                                micro_has_valid = bool(policy_mask.any().item())
                                accumulated_has_valid |= micro_has_valid

                                output, vpred_temp = ppo_impl.forward(
                                    model,
                                    mb_query_responses,
                                    processing_class.pad_token_id,
                                )
                                logits = output.logits[:, context_length - 1 : -1]
                                logits /= args.temperature + 1e-7
                                new_logprobs = selective_log_softmax_token_chunks(
                                    logits, mb_responses
                                )
                                new_logprobs = torch.masked_fill(
                                    new_logprobs,
                                    ~policy_mask,
                                    ppo_impl.INVALID_LOGPROB,
                                )
                                vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                                vpred = torch.masked_fill(vpred, ~value_mask, 0)
                                vpredclipped = torch.clamp(
                                    vpred,
                                    mb_values - args.cliprange_value,
                                    mb_values + args.cliprange_value,
                                )
                                vf_losses1 = torch.square(vpred - mb_return)
                                vf_losses2 = torch.square(vpredclipped - mb_return)
                                vf_loss_max = torch.max(vf_losses1, vf_losses2)
                                vf_loss = 0.5 * masked_mean_or_zero(
                                    vf_loss_max, value_mask
                                )
                                vf_clipfrac = masked_mean_or_zero(
                                    (vf_losses2 > vf_losses1).float(), value_mask
                                )
                                logprobs_diff = new_logprobs - mb_logprobs
                                ratio = torch.exp(logprobs_diff)
                                pg_losses = -mb_advantage * ratio
                                pg_losses2 = -mb_advantage * torch.clamp(
                                    ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                                )
                                pg_loss_max = torch.max(pg_losses, pg_losses2)
                                pg_loss = masked_mean_or_zero(pg_loss_max, policy_mask)
                                loss = pg_loss + args.vf_coef * vf_loss
                                accelerator.backward(loss)
                                # A synchronized all-invalid accumulation window
                                # must not even apply AdamW weight decay.
                                if not accelerator.sync_gradients or accumulated_has_valid:
                                    optimizer.step()
                                optimizer.zero_grad()
                                if accelerator.sync_gradients:
                                    accumulated_has_valid = False

                                with torch.no_grad():
                                    pg_clipfrac = masked_mean_or_zero(
                                        (pg_losses2 > pg_losses).float(), policy_mask
                                    )
                                    entropy = entropy_from_logits_token_chunks(logits)
                                    approxkl = 0.5 * masked_mean_or_zero(
                                        logprobs_diff**2, policy_mask
                                    )
                                    stat_index = (
                                        ppo_epoch_idx,
                                        minibatch_idx,
                                        gradient_accumulation_idx,
                                    )
                                    valid_stat_mask[stat_index] = micro_has_valid
                                    approxkl_stats[stat_index] = approxkl
                                    pg_clipfrac_stats[stat_index] = pg_clipfrac
                                    pg_loss_stats[stat_index] = pg_loss
                                    vf_loss_stats[stat_index] = vf_loss
                                    vf_clipfrac_stats[stat_index] = vf_clipfrac
                                    entropy_stats[stat_index] = masked_mean_or_zero(
                                        entropy, policy_mask
                                    )
                                    ratio_stats[stat_index] = masked_mean_or_zero(
                                        ratio, policy_mask
                                    )
                            gradient_accumulation_idx += 1
                        minibatch_idx += 1
                        ppo_impl.empty_cache()
            else:
                optimizer.zero_grad(set_to_none=True)
                accelerator.print(
                    f"[drift] PPO batch {update}: all {n_scheduled} completions "
                    "were truncated; optimizer update skipped"
                )

            with torch.no_grad():
                token_mask = ~padding_mask
                valid_scores = scores[terminated]
                mean_kl = masked_mean_or_zero(kl.sum(1), terminated)
                mean_entropy = masked_mean_or_zero((-logprobs).sum(1), terminated)
                mean_non_score_reward = masked_mean_or_zero(
                    non_score_reward.sum(1), terminated
                )
                mean_score = (
                    valid_scores.mean() if valid_scores.numel() else scores.sum() * 0
                )
                rlhf_reward = mean_non_score_reward + mean_score
                eps = int(self.state.episode / max(time.time() - start_time, 1e-9))
                metric_mean = lambda value: masked_mean_or_zero(value, valid_stat_mask)
                ratio_count = int(valid_stat_mask.sum().item())
                ratio_mean = metric_mean(ratio_stats)
                ratio_var = (
                    ((ratio_stats[valid_stat_mask] - ratio_mean) ** 2).mean()
                    if ratio_count > 1
                    else ratio_mean * 0
                )
                metrics = {
                    "eps": eps,
                    "objective/kl": mean_kl.item(),
                    "objective/entropy": mean_entropy.item(),
                    "objective/non_score_reward": mean_non_score_reward.item(),
                    "objective/rlhf_reward": rlhf_reward.item(),
                    "objective/scores": mean_score.item(),
                    "objective/termination_rate": n_terminated / max(n_scheduled, 1),
                    "objective/num_terminated": n_terminated,
                    "objective/num_truncated": n_truncated,
                    "objective/effective_updates": self.completion_masking_stats[
                        "effective_updates"
                    ],
                    "policy/approxkl_avg": metric_mean(approxkl_stats).item(),
                    "policy/clipfrac_avg": metric_mean(pg_clipfrac_stats).item(),
                    "loss/policy_avg": metric_mean(pg_loss_stats).item(),
                    "loss/value_avg": metric_mean(vf_loss_stats).item(),
                    "val/clipfrac_avg": metric_mean(vf_clipfrac_stats).item(),
                    "policy/entropy_avg": metric_mean(entropy_stats).item(),
                    "val/ratio": ratio_mean.item(),
                    "val/ratio_var": ratio_var.item(),
                    "val/num_eos_tokens": sum(
                        int((responses == token_id).sum().item())
                        for token_id in self.termination_eos_token_ids
                    ),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "episode": self.state.episode,
                }
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(
                args, self.state, self.control
            )
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )
            del scores, metrics, non_score_reward
            ppo_impl.empty_cache()
            gc.collect()

            if (
                args.num_sample_generations > 0
                and (update - 1) % self.sample_generations_freq == 0
            ):
                self.generate_completions(sampling=True)
                ppo_impl.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                terminated,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            ppo_impl.empty_cache()

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
