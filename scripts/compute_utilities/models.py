# models.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import random
from .diagnostics import edge_diagnostics
from .utils import has_multiple_answers

class UtilityModel(ABC):
    """
    Abstract base class for utility models that learn preferences from pairwise comparisons.
    
    Now stateful, with a 'unparseable_mode' attribute controlling how to handle unparseable responses.
    """

    def __init__(
        self,
        unparseable_mode: str,
        comparison_prompt_template: str,
        system_message: str,
        with_reasoning: bool,
        terminal_only: bool = False,
        label_choices: Optional[List[str]] = None,
        raw_dump_path: Optional[str] = None,
        raw_dump_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize a UtilityModel.
        
        Args:
            unparseable_mode: How to handle unparseable responses (can be "skip", "random", or "distribution")
            comparison_prompt_template: Template for comparison prompts
            system_message: System message for agents that accept a system message
            with_reasoning: Whether to use response parsing for comparison_prompt_template_reasoning_default
            label_choices: Two labels used for the first and second presented options
            raw_dump_path: Optional JSONL path for raw sampled responses
            raw_dump_metadata: Metadata to copy onto each raw-dump row
            **kwargs: Additional arguments specific to each utility model implementation
        """
        # Validate unparseable_mode
        valid_modes = ["skip", "random", "distribution"]
        if unparseable_mode not in valid_modes:
            raise ValueError(f"unparseable_mode must be one of {valid_modes}, got '{unparseable_mode}'")
        
        # Store required arguments as attributes
        self.unparseable_mode = unparseable_mode
        self.comparison_prompt_template = comparison_prompt_template
        self.system_message = system_message
        self.with_reasoning = with_reasoning
        self.terminal_only = terminal_only
        self.label_choices = label_choices or ["A", "B"]
        if len(self.label_choices) != 2:
            raise ValueError("label_choices must contain exactly two labels")
        if not all(isinstance(label, str) and label.strip() for label in self.label_choices):
            raise ValueError("label_choices must be non-empty strings")
        if self.label_choices[0].lower() == self.label_choices[1].lower():
            raise ValueError("label_choices must be distinct")
        self.raw_dump_path = raw_dump_path
        self.raw_dump_metadata = raw_dump_metadata or {}

    def write_raw_response_dump(
        self,
        graph: 'PreferenceGraph',
        prompts: List[str],
        responses: Dict[int, List[str]],
        parsed_responses: Dict[int, List[str]],
        prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
        split: str,
        iteration: Optional[int] = None,
    ) -> None:
        """Append raw sampled completions to a JSONL sidecar."""
        if not self.raw_dump_path:
            return
        os.makedirs(os.path.dirname(self.raw_dump_path) or ".", exist_ok=True)
        with open(self.raw_dump_path, "a") as f:
            for prompt_idx, raw_list in responses.items():
                A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
                option_A = graph.options_by_id[A_id]
                option_B = graph.options_by_id[B_id]
                if direction == "flipped":
                    presented_first = option_B
                    presented_second = option_A
                else:
                    presented_first = option_A
                    presented_second = option_B
                parsed_list = parsed_responses.get(prompt_idx, [])
                for sample_idx, raw_response in enumerate(raw_list):
                    parsed = parsed_list[sample_idx] if sample_idx < len(parsed_list) else "unparseable"
                    row = {
                        **self.raw_dump_metadata,
                        "split": split,
                        "iteration": iteration,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "option_a_id": A_id,
                        "option_b_id": B_id,
                        "option_a_description": option_A["description"],
                        "option_b_description": option_B["description"],
                        "direction": direction,
                        "presented_first_id": presented_first["id"],
                        "presented_second_id": presented_second["id"],
                        "presented_first_description": presented_first["description"],
                        "presented_second_description": presented_second["description"],
                        "prompt": prompts[prompt_idx],
                        "raw_response": raw_response,
                        "parsed_response": parsed,
                        "parse_status": "parsed" if parsed in self.label_choices else "unparseable",
                        "multiple_answers": has_multiple_answers(raw_response, self.label_choices),
                        "parsed_response_semantics": "presented_label",
                    }
                    f.write(json.dumps(row) + "\n")

    @abstractmethod
    async def fit(
        self,
        graph: 'PreferenceGraph',
        agent: Any
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
        """
        Fit the utility model to the preference data.
        
        Args:
            graph: PreferenceGraph object containing the preference data
            agent: The agent used for generating comparisons
            
        Returns:
            Tuple containing:
            - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            - metrics: Dict containing model metrics like log_loss and accuracy
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        graph: 'PreferenceGraph',
        utilities: Dict[Any, Dict[str, float]],
        edge_indices: List[Tuple[Any, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate the model's goodness-of-fit on the given edges.
        
        Args:
            graph: PreferenceGraph object containing the preference data
            utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            edge_indices: List of (option_A_id, option_B_id) tuples to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics (e.g. log_loss, accuracy)
        """
        pass

    def process_responses(
        self,
        graph: 'PreferenceGraph',
        responses: Dict[int, List[str]],
        parsed_responses: Dict[int, List[str]],
        prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
    ) -> List[Dict]:
        """
        Convert raw responses into probabilities of preferring A over B.

        The logic is equivalent to the original PreferenceGraph.process_responses,
        **but** now we incorporate `unparseable_mode` handling:
           - "skip": ignore unparseable
           - "random": randomly choose A or B for unparseable
           - "distribution": treat unparseable as [0.5, 0.5]

        Args:
            graph: The PreferenceGraph containing options and edges
            responses: Dict mapping prompt_idx to list of K raw responses
            parsed_responses: Dict mapping prompt_idx to list of K parsed responses ('A', 'B', or 'unparseable')
            prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)

        Returns:
            A list of preference data dictionaries ready for `graph.add_edges`, where each entry has:
              - option_A
              - option_B
              - probability_A
              - aux_data (with counts, original responses, etc.)
        """
        # Group raw responses by pair (in the original orientation)
        pair_data = {}  # (A_id, B_id) -> data structure

        for prompt_idx, response_list in responses.items():
            A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
            parsed_list = parsed_responses[prompt_idx]  # The K parsed responses

            # Use the orientation as-is (A_id, B_id)
            pair_key = (A_id, B_id)
            if pair_key not in pair_data:
                pair_data[pair_key] = {
                    'option_A': graph.options_by_id[A_id],
                    'option_B': graph.options_by_id[B_id],
                    'original_responses': [],
                    'flipped_responses': [],
                    'original_parsed': [],
                    'flipped_parsed': []
                }

            # We store the raw and parsed responses in separate buckets
            if direction == 'original':
                pair_data[pair_key]['original_responses'].extend(response_list)
                pair_data[pair_key]['original_parsed'].extend(parsed_list)
            else:  # 'flipped'
                pair_data[pair_key]['flipped_responses'].extend(response_list)
                pair_data[pair_key]['flipped_parsed'].extend(parsed_list)

        # Now, we convert each pair's responses into a probability P(A)
        preference_data = []
        rng = random.Random(42)  # or some other seed for stable 'random' in "random" mode

        for (A_id, B_id), data in pair_data.items():
            # Instead of just counting 'A' or 'B', we keep track of distributions:
            # e.g. 'A' -> (1,0), 'B' -> (0,1), 'unparseable' -> depends on unparseable_mode.
            dist_list = []  # Will store (a_val, b_val) for each response

            # A small helper for flipping:
            # if direction == 'flipped' and user picks 'A', that means B in original orientation
            # if direction == 'flipped' and user picks 'B', that means A in original orientation
            first_label, second_label = self.label_choices

            def add_to_dist_list(parsed_char, is_flipped=False):
                """
                Add an appropriate (A, B) distribution to dist_list based on the parse,
                flipping if needed.
                """
                if parsed_char == first_label:
                    if not is_flipped:
                        dist_list.append((1.0, 0.0))
                    else:
                        dist_list.append((0.0, 1.0))
                elif parsed_char == second_label:
                    if not is_flipped:
                        dist_list.append((0.0, 1.0))
                    else:
                        dist_list.append((1.0, 0.0))
                else:
                    # unparseable
                    if self.unparseable_mode == "skip":
                        # do nothing -> skip
                        pass
                    elif self.unparseable_mode == "random":
                        # randomly pick A or B
                        if rng.random() < 0.5:
                            # pick A
                            if not is_flipped:
                                dist_list.append((1.0, 0.0))
                            else:
                                dist_list.append((0.0, 1.0))
                        else:
                            # pick B
                            if not is_flipped:
                                dist_list.append((0.0, 1.0))
                            else:
                                dist_list.append((1.0, 0.0))
                    elif self.unparseable_mode == "distribution":
                        # treat as 50/50
                        dist_list.append((0.5, 0.5))

            # Original ordering
            for parsed in data['original_parsed']:
                add_to_dist_list(parsed, is_flipped=False)

            # Flipped ordering
            for parsed in data['flipped_parsed']:
                add_to_dist_list(parsed, is_flipped=True)

            # Summarize results
            total_A = sum(d[0] for d in dist_list)
            total_B = sum(d[1] for d in dist_list)
            total_responses = len(dist_list)

            if total_responses > 0:
                probability_A = total_A / (total_A + total_B)  # or total_A / total_responses if we interpret it differently
                aux_data = {
                    'count_A': total_A,  # This might be fractional now
                    'count_B': total_B,  # Also might be fractional
                    'total_responses': total_responses,
                    'original_responses': data['original_responses'],
                    'flipped_responses': data['flipped_responses'],
                    'original_parsed': data['original_parsed'],
                    'flipped_parsed': data['flipped_parsed'],
                    'unparseable_mode': self.unparseable_mode,
                    'label_choices': self.label_choices
                }
                aux_data['diagnostics'] = edge_diagnostics(aux_data, probability_A)
                entry = {
                    'option_A': data['option_A'],
                    'option_B': data['option_B'],
                    'probability_A': probability_A,
                    'aux_data': aux_data
                }
                preference_data.append(entry)
            else:
                # if total_responses == 0, that means everything was "skip"
                # or no valid responses were found
                pass

        return preference_data

    def process_choice_probs(
        self,
        graph: 'PreferenceGraph',
        choice_probs: Dict[int, Optional[float]],
        prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
    ) -> List[Dict]:
        """
        Logprobs-mode parallel of process_responses.

        Each prompt contributes its model-implied P_A directly as fractional
        ``(count_A, count_B)`` rather than integer counts over K hard samples:
          - original orientation: ``(P_A, 1 - P_A)``
          - flipped  orientation: ``(1 - P_A, P_A)``  (the flipped prompt
            asked (B, A), so the model's P(first option) is P(B) in the
            original orientation).

        With include_flipped=True, each edge ends up with total weight 2 (one
        per orientation). The resulting fractional counts flow through the
        existing ``probability_A = count_A / (count_A + count_B)`` machinery
        unchanged.

        Unparseable prompts (P_A is None) are handled per
        ``self.unparseable_mode``, mirroring process_responses:
          - "skip":         not added
          - "random":       (1, 0) or (0, 1) with 50/50 chance
          - "distribution": (0.5, 0.5)
        """
        rng = random.Random(42)
        pair_data = {}
        for prompt_idx, p_a in choice_probs.items():
            A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
            pair_key = (A_id, B_id)
            d = pair_data.setdefault(pair_key, {
                'option_A': graph.options_by_id[A_id],
                'option_B': graph.options_by_id[B_id],
                'count_A': 0.0,
                'count_B': 0.0,
                'total': 0.0,
                'choice_probs': [],
            })
            d['choice_probs'].append({'direction': direction, 'p_a': p_a})

            if p_a is None:
                if self.unparseable_mode == 'skip':
                    continue
                if self.unparseable_mode == 'random':
                    p_a = 1.0 if rng.random() < 0.5 else 0.0
                elif self.unparseable_mode == 'distribution':
                    p_a = 0.5
                else:
                    continue

            if direction == 'flipped':
                a_contrib, b_contrib = 1.0 - p_a, p_a
            else:
                a_contrib, b_contrib = p_a, 1.0 - p_a
            d['count_A'] += a_contrib
            d['count_B'] += b_contrib
            d['total'] += 1.0

        preference_data = []
        for (A_id, B_id), d in pair_data.items():
            if d['total'] <= 0:
                continue
            probability_A = d['count_A'] / (d['count_A'] + d['count_B'])
            aux_data = {
                'count_A': d['count_A'],
                'count_B': d['count_B'],
                'total_responses': d['total'],
                'choice_probs': d['choice_probs'],
                'unparseable_mode': self.unparseable_mode,
                'mode': 'logprobs',
                'label_choices': self.label_choices,
            }
            aux_data['diagnostics'] = edge_diagnostics(aux_data, probability_A)
            preference_data.append({
                'option_A': d['option_A'],
                'option_B': d['option_B'],
                'probability_A': probability_A,
                'aux_data': aux_data,
            })
        return preference_data
