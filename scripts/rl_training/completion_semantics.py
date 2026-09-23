"""Authoritative completion-termination semantics shared by online RL arms.

A rollout that reaches the generation cap without emitting an EOS token is not
a wrong answer: it is an unobserved/unfinished answer.  Such a rollout must not
enter reward normalization, advantage estimation, or a policy/value loss.
"""

from __future__ import annotations

from collections.abc import Iterable


def normalize_eos_token_ids(*sources) -> tuple[int, ...]:
    """Return a stable, de-duplicated tuple of EOS ids from scalar/list sources."""
    ids: list[int] = []
    for source in sources:
        if source is None:
            continue
        values = source if isinstance(source, (list, tuple, set)) else [source]
        for value in values:
            token_id = int(value)
            if token_id not in ids:
                ids.append(token_id)
    return tuple(ids)


def model_eos_token_ids(model, tokenizer) -> tuple[int, ...]:
    """Collect every legitimate model/tokenizer generation terminator."""
    generation_ids = getattr(
        getattr(model, "generation_config", None), "eos_token_id", None
    )
    config_ids = getattr(getattr(model, "config", None), "eos_token_id", None)
    tokenizer_ids = getattr(tokenizer, "eos_token_id", None)
    ids = normalize_eos_token_ids(generation_ids, config_ids, tokenizer_ids)
    if not ids:
        raise ValueError("no EOS token id is configured for completion termination")
    return ids


def completion_ids_terminated(
    completion_ids: Iterable[int], eos_token_ids: Iterable[int]
) -> bool:
    """Whether an unpadded generated completion emitted an authoritative EOS."""
    eos_ids = set(normalize_eos_token_ids(eos_token_ids))
    if not eos_ids:
        raise ValueError("eos_token_ids must not be empty")
    # Generation stops at EOS, but checking membership is robust to callers that
    # retain right-padding or other terminal bookkeeping after that token.
    return any(int(token_id) in eos_ids for token_id in completion_ids)


def tensor_completion_termination_mask(responses, eos_token_ids):
    """Vectorized EOS-membership mask for a ``[batch, generated_tokens]`` tensor."""
    import torch

    if responses.ndim != 2:
        raise ValueError(f"responses must be rank 2, got shape={tuple(responses.shape)}")
    eos_ids = normalize_eos_token_ids(eos_token_ids)
    if not eos_ids:
        raise ValueError("eos_token_ids must not be empty")
    terminated = torch.zeros(responses.shape[0], dtype=torch.bool, device=responses.device)
    for token_id in eos_ids:
        terminated |= torch.any(responses == token_id, dim=-1)
    return terminated


def truncate_responses_at_eos(responses, eos_token_ids, pad_token_id: int):
    """Keep the first EOS (from any configured id) and pad everything after it."""
    import torch

    eos_ids = normalize_eos_token_ids(eos_token_ids)
    if not eos_ids:
        raise ValueError("eos_token_ids must not be empty")
    is_eos = torch.zeros_like(responses, dtype=torch.bool)
    for token_id in eos_ids:
        is_eos |= responses == token_id
    first_eos = torch.argmax(is_eos.to(torch.int64), dim=-1)
    has_eos = torch.any(is_eos, dim=-1)
    positions = torch.arange(responses.shape[1], device=responses.device).unsqueeze(0)
    after_eos = has_eos.unsqueeze(1) & (positions > first_eos.unsqueeze(1))
    return torch.masked_fill(responses, after_eos, int(pad_token_id))
