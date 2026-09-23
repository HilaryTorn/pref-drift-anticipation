"""Diagnostics for forced-choice preference elicitation runs."""

from __future__ import annotations

from statistics import mean
from typing import Any

from .utils import has_multiple_answers


# NOTE: there is deliberately no refusal metric here. It was a substring match for
# cannot|can't|won't|unable|sorry|refuse|... anywhere in the response, which on a reasoning-on run
# fires on ordinary deliberation ("Person B can't keep track...", "I won't have autonomy...") rather
# than an actual refusal. On the 4B value-pairs battery it flagged 97% of samples (523/540) as
# refusals -- pure noise -- so it was removed rather than left to mislead. These base models don't
# refuse a "which would you rather be" choice; if a genuine refusal signal is ever needed, judge it,
# don't substring-match it.

POSITION_STRATEGY_THRESHOLD = 0.95
TIE_EPS = 1e-9


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _hard_choice_diagnostics(aux: dict[str, Any], probability_a: float) -> dict[str, Any]:
    label_choices = aux.get("label_choices", ["A", "B"])
    first_label, second_label = label_choices
    valid_label_set = set(label_choices)
    original = aux.get("original_parsed", [])
    flipped = aux.get("flipped_parsed", [])
    original_raw = aux.get("original_responses", [])
    flipped_raw = aux.get("flipped_responses", [])

    all_parsed = original + flipped
    all_raw = original_raw + flipped_raw
    valid = [label for label in all_parsed if label in valid_label_set]
    valid_count = len(valid)
    raw_count = len(all_parsed)
    unparseable_count = sum(1 for label in all_parsed if label == "unparseable")
    none_count = sum(1 for response in all_raw if response is None)
    # Take-last picks the final "Answer:" line; this counts samples whose answer lines disagreed, so
    # the committed label was a guess between a revision and a contradiction. Reported, not dropped.
    multiple_answer_count = sum(1 for response in all_raw if has_multiple_answers(response, label_choices))

    original_valid = [label for label in original if label in valid_label_set]
    flipped_valid = [label for label in flipped if label in valid_label_set]

    original_underlying_a = sum(1 for label in original_valid if label == first_label)
    flipped_underlying_a = sum(1 for label in flipped_valid if label == second_label)
    original_underlying_a_rate = _safe_rate(original_underlying_a, len(original_valid))
    flipped_underlying_a_rate = _safe_rate(flipped_underlying_a, len(flipped_valid))
    order_effect_gap = None
    if original_underlying_a_rate is not None and flipped_underlying_a_rate is not None:
        order_effect_gap = original_underlying_a_rate - flipped_underlying_a_rate

    first_label_count = sum(1 for label in valid if label == first_label)
    second_label_count = sum(1 for label in valid if label == second_label)
    position_first_label_rate = _safe_rate(first_label_count, valid_count)
    position_first_label_bias = None if position_first_label_rate is None else position_first_label_rate - 0.5

    always_pick_label = None
    if original_valid and flipped_valid and valid_count:
        first_label_rate = first_label_count / valid_count
        second_label_rate = second_label_count / valid_count
        if first_label_rate >= POSITION_STRATEGY_THRESHOLD:
            always_pick_label = first_label
        elif second_label_rate >= POSITION_STRATEGY_THRESHOLD:
            always_pick_label = second_label

    degenerate_tie = always_pick_label is not None and abs(probability_a - 0.5) <= TIE_EPS

    return {
        "mode": "hard_samples",
        "label_choices": label_choices,
        "raw_response_count": raw_count,
        "valid_response_count": valid_count,
        "unparseable_count": unparseable_count,
        "none_response_count": none_count,
        "invalid_count": unparseable_count,
        "invalid_rate": _safe_rate(unparseable_count, raw_count),
        "multiple_answer_count": multiple_answer_count,
        "multiple_answer_rate": _safe_rate(multiple_answer_count, raw_count),
        "position_label_counts": {first_label: first_label_count, second_label: second_label_count},
        "position_first_label_rate": position_first_label_rate,
        "position_first_label_bias": position_first_label_bias,
        "position_a_rate": position_first_label_rate if first_label == "A" else None,
        "position_a_bias": position_first_label_bias if first_label == "A" else None,
        "original_underlying_a_rate": original_underlying_a_rate,
        "flipped_underlying_a_rate": flipped_underlying_a_rate,
        "order_effect_gap": order_effect_gap,
        "always_pick_label": always_pick_label,
        "degenerate_tie": degenerate_tie,
    }


def _logprobs_diagnostics(aux: dict[str, Any], probability_a: float) -> dict[str, Any]:
    label_choices = aux.get("label_choices", ["A", "B"])
    first_label, second_label = label_choices
    choice_probs = aux.get("choice_probs", [])
    prompt_count = len(choice_probs)
    none_count = sum(1 for item in choice_probs if item.get("p_a") is None)
    valid_items = [item for item in choice_probs if item.get("p_a") is not None]

    original_probs = [
        item["p_a"]
        for item in valid_items
        if item.get("direction") == "original"
    ]
    flipped_presented_a_probs = [
        item["p_a"]
        for item in valid_items
        if item.get("direction") == "flipped"
    ]
    flipped_underlying_a_probs = [1.0 - p for p in flipped_presented_a_probs]

    original_underlying_a_rate = mean(original_probs) if original_probs else None
    flipped_underlying_a_rate = mean(flipped_underlying_a_probs) if flipped_underlying_a_probs else None
    order_effect_gap = None
    if original_underlying_a_rate is not None and flipped_underlying_a_rate is not None:
        order_effect_gap = original_underlying_a_rate - flipped_underlying_a_rate

    presented_a_probs = [item["p_a"] for item in valid_items]
    position_first_label_rate = mean(presented_a_probs) if presented_a_probs else None
    position_first_label_bias = None if position_first_label_rate is None else position_first_label_rate - 0.5

    always_pick_label = None
    if original_probs and flipped_presented_a_probs:
        all_presented_a = all(p >= POSITION_STRATEGY_THRESHOLD for p in presented_a_probs)
        all_presented_b = all(p <= 1.0 - POSITION_STRATEGY_THRESHOLD for p in presented_a_probs)
        if all_presented_a:
            always_pick_label = first_label
        elif all_presented_b:
            always_pick_label = second_label

    degenerate_tie = always_pick_label is not None and abs(probability_a - 0.5) <= TIE_EPS

    return {
        "mode": "logprobs",
        "label_choices": label_choices,
        "prompt_count": prompt_count,
        "valid_prompt_count": len(valid_items),
        "none_choice_prob_count": none_count,
        "invalid_count": none_count,
        "invalid_rate": _safe_rate(none_count, prompt_count),
        "position_first_label_rate": position_first_label_rate,
        "position_first_label_bias": position_first_label_bias,
        "position_a_rate": position_first_label_rate if first_label == "A" else None,
        "position_a_bias": position_first_label_bias if first_label == "A" else None,
        "original_underlying_a_rate": original_underlying_a_rate,
        "flipped_underlying_a_rate": flipped_underlying_a_rate,
        "order_effect_gap": order_effect_gap,
        "always_pick_label": always_pick_label,
        "degenerate_tie": degenerate_tie,
    }


def edge_diagnostics(aux: dict[str, Any], probability_a: float) -> dict[str, Any]:
    if aux.get("mode") == "logprobs" or "choice_probs" in aux:
        return _logprobs_diagnostics(aux, probability_a)
    return _hard_choice_diagnostics(aux, probability_a)


def summarize_graph_diagnostics(graph: Any) -> dict[str, Any]:
    edges = list(graph.edges.values())
    diagnostics = [edge.aux_data.get("diagnostics", {}) for edge in edges]

    def count_edges(key: str, value: Any = True) -> int:
        return sum(1 for item in diagnostics if item.get(key) == value)

    invalid_counts = [item.get("invalid_count", 0) or 0 for item in diagnostics]
    multiple_answer_counts = [item.get("multiple_answer_count", 0) or 0 for item in diagnostics]
    prompt_counts = [
        item.get("raw_response_count", item.get("prompt_count", 0)) or 0
        for item in diagnostics
    ]
    order_gaps = [
        abs(item["order_effect_gap"])
        for item in diagnostics
        if item.get("order_effect_gap") is not None
    ]
    position_biases = [
        abs(item["position_first_label_bias"])
        for item in diagnostics
        if item.get("position_first_label_bias") is not None
    ]
    label_choices = next(
        (item.get("label_choices") for item in diagnostics if item.get("label_choices")),
        ["A", "B"],
    )

    total_prompts = sum(prompt_counts)
    total_invalid = sum(invalid_counts)
    total_multiple_answers = sum(multiple_answer_counts)

    return {
        "label_scheme": f"{label_choices[0]}/{label_choices[1]}",
        "label_choices": label_choices,
        "order_normalization": "original_and_flipped",
        "position_strategy_threshold": POSITION_STRATEGY_THRESHOLD,
        "edge_count": len(edges),
        "total_prompt_or_response_count": total_prompts,
        "invalid_count": total_invalid,
        "invalid_rate": _safe_rate(total_invalid, total_prompts),
        "multiple_answer_count": total_multiple_answers,
        "multiple_answer_rate": _safe_rate(total_multiple_answers, total_prompts),
        "edges_with_invalid": sum(1 for count in invalid_counts if count > 0),
        "always_pick_first_label_edges": count_edges("always_pick_label", label_choices[0]),
        "always_pick_second_label_edges": count_edges("always_pick_label", label_choices[1]),
        "always_pick_a_edges": count_edges("always_pick_label", "A"),
        "always_pick_b_edges": count_edges("always_pick_label", "B"),
        "degenerate_tie_edges": count_edges("degenerate_tie", True),
        "mean_abs_order_effect_gap": mean(order_gaps) if order_gaps else None,
        "mean_abs_position_first_label_bias": mean(position_biases) if position_biases else None,
        "mean_abs_position_a_bias": (
            mean(position_biases) if label_choices[0] == "A" and position_biases else None
        ),
        "label_scheme_robustness": "not_run",
    }
