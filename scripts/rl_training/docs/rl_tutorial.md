# RL Algorithm Tutorial

This note explains the RL algorithms used in the drift experiment and how their
components map onto the current implementation.

The experiment is not human-preference RLHF. It is verifier-based coding RL:
the model receives a programming prompt, generates code, and the reward comes
from executable tests.

## Shared Setup

Let:

- `x`: coding prompt/problem statement.
- `y`: model completion/code solution.
- `pi_theta(y | x)`: trainable policy model.
- `pi_ref(y | x)`: frozen reference policy, usually the base model.
- `r(x, y)`: verifier reward, currently unit-test pass rate in `[0, 1]`.
- `beta`: KL/reference regularization strength.

The shared reward is:

```text
r(x, y) = pass_rate(y; verifier_x)
```

For `N` tests:

```text
pass_rate(y) = (1 / N) * sum_i 1[test_i(y) passes]
```

The high-level regularized objective for online RL is:

```text
maximize  E_{x, y ~ pi_theta}[ r(x, y) ]
          - beta * KL( pi_theta(. | x) || pi_ref(. | x) )
```

The KL term is the drift-budget control: it rewards solving the coding task but
penalizes moving too far from the reference model.

In this repo:

- Data rows are `coding_task_v1` with `prompt` and `verifier`.
- `training/prompts.py` formats raw prompts into Python stdin/stdout prompts.
- `training/rewards.py` computes `r(x, y)` through `run_verifier`.
- `training/train_rl.py` wires the algorithms through TRL.
- `training/configs/*.yaml` sets matched hyperparameters such as `max_steps`,
  `save_steps`, learning rate, batch size, and `beta` / `init_kl_coef`.
- LoRA is available through `--use_lora`. In the main experiment, LoRA should be
  held fixed across algorithms; it is a parameter-efficient training substrate,
  not the experimental variable.

## PPO

PPO is an online policy-gradient method. It samples completions from the current
policy, scores them, estimates an advantage, and updates the policy while
clipping large policy-ratio changes.

### Components

- **Policy model** `pi_theta`: the model being trained.
- **Reference model** `pi_ref`: frozen base model used for KL control.
- **Reward model / reward source**: in standard TRL PPO this is a reward model.
- **Value model** `V_phi(x, y_<t)`: estimates expected return for variance
  reduction.
- **Advantage** `A_t`: how much better a sampled token/action was than expected.
- **Clipping coefficient** `epsilon`: prevents overly large policy updates.
- **KL coefficient** `beta` or `kl_coef`: penalizes divergence from the
  reference policy.

### Objective

For token-level ratio:

```text
rho_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

PPO uses the clipped surrogate:

```text
L_policy(theta) =
  E_t[ min(
    rho_t(theta) * A_t,
    clip(rho_t(theta), 1 - epsilon, 1 + epsilon) * A_t
  ) ]
```

The reward is usually KL-regularized:

```text
R_t = r(x, y) - beta * log( pi_theta(y_t | x, y_<t) / pi_ref(y_t | x, y_<t) )
```

The value model is trained with a regression loss:

```text
L_value(phi) = E_t[ ( V_phi(s_t) - R_t^target )^2 ]
```

The total PPO optimization is approximately:

```text
maximize L_policy(theta) - c_v * L_value(phi) + entropy_bonus
```

### In This Repo

The current `train_ppo` path uses TRL's `PPOTrainer`, which expects a reward
model, not a Python reward function. Therefore PPO requires:

```bash
--reward_model <reward-model-path>
```

To keep PPO comparable to GRPO/DPO, that reward model must be trained from the
same verifier signal used elsewhere: unit-test pass rate or preference pairs
derived from unit-test pass rate. Running PPO with an unrelated reward model
would change the objective and confound the algorithm comparison.

Relevant config: `training/configs/ppo.yaml`

Key fields:

- `init_kl_coef`: KL/reference penalty, matched to GRPO/DPO `beta`.
- `num_ppo_epochs`: PPO update epochs per batch.
- `response_length`: generation length.
- `max_steps`, `save_steps`: optimizer budget and checkpoint cadence.

## GRPO

GRPO, Group Relative Policy Optimization, is an online RL method designed to
avoid a separate learned value model. Instead of estimating a baseline with
`V_phi`, it samples a group of completions for the same prompt and normalizes
each completion's reward relative to the group.

### Components

- **Policy model** `pi_theta`: the model being trained.
- **Reference model** `pi_ref`: implicit/frozen reference used by TRL for KL.
- **Reward function** `r(x, y)`: Python verifier reward in this repo.
- **Group size** `G`: number of completions sampled for each prompt.
- **Group-relative advantage** `A_i`: normalized reward within a prompt group.
- **KL coefficient** `beta`: penalizes divergence from the reference policy.

### Group Advantage

For one prompt `x`, sample `G` completions:

```text
y_1, ..., y_G ~ pi_theta(. | x)
```

Score them:

```text
r_i = r(x, y_i)
```

Compute group mean and standard deviation:

```text
mu = (1 / G) * sum_i r_i
sigma = std(r_1, ..., r_G)
```

The relative advantage is commonly:

```text
A_i = (r_i - mu) / (sigma + eps)
```

This says: a completion is good if it is better than the other completions from
the same prompt, not merely high in absolute reward.

### Objective

GRPO uses a policy-ratio objective similar in spirit to PPO, but with the
group-relative advantage and no separate value loss:

```text
maximize E_i[
  policy_ratio_i(theta) * A_i
  - beta * KL( pi_theta(. | x) || pi_ref(. | x) )
]
```

TRL's exact implementation details may include clipping and token-level
normalization, but the key distinction is: baseline comes from the sampled group,
not from a learned value model.

### In This Repo

GRPO is the cleanest online verifier-RL path because it directly accepts the
Python reward function:

```python
reward_funcs=[coding_reward]
```

Relevant config: `training/configs/grpo.yaml`

Key fields:

- `num_generations`: group size `G`.
- `beta`: KL/reference penalty.
- `max_completion_length`: generation length.
- `reward_timeout`: verifier timeout.
- `eval_reward_prompts`, `eval_reward_samples`: per-checkpoint reward logging.

## DPO

DPO, Direct Preference Optimization, is an offline preference-optimization
algorithm. It does not call a reward function during training. Instead, it
trains on preference pairs:

```text
(x, y_w, y_l)
```

where `y_w` is the chosen/winning answer and `y_l` is the rejected/losing answer.

In this project, DPO pairs are generated from the same verifier used by online
RL:

```text
y_w = solution with higher pass_rate
y_l = solution with lower pass_rate
```

Pairs are kept only when the reward margin is large enough, for example:

```text
r(x, y_w) - r(x, y_l) >= 0.5
```

### Components

- **Policy model** `pi_theta`: the model being trained.
- **Reference model** `pi_ref`: frozen base model.
- **Chosen completion** `y_w`: verifier-preferred solution.
- **Rejected completion** `y_l`: lower-scoring solution.
- **Preference temperature / KL parameter** `beta`: controls how strongly the
  model moves away from the reference policy.

### Objective

DPO starts from a KL-regularized reward-model view, but avoids explicitly
training a reward model. The loss for one preference pair is:

```text
L_DPO(theta) =
  - log sigmoid(
      beta * [
        log pi_theta(y_w | x) - log pi_ref(y_w | x)
        - log pi_theta(y_l | x) + log pi_ref(y_l | x)
      ]
    )
```

Equivalently:

```text
L_DPO(theta) =
  - log sigmoid(
      beta * [
        log( pi_theta(y_w | x) / pi_ref(y_w | x) )
        - log( pi_theta(y_l | x) / pi_ref(y_l | x) )
      ]
    )
```

The model is rewarded when it increases the chosen answer's likelihood relative
to the rejected answer, after correcting for what the reference model already
preferred.

### In This Repo

DPO data is produced by:

```bash
python training/build_dpo_pairs.py \
  --model <base-or-sft-model> \
  --prompts <coding_task_v1_train.jsonl> \
  --out <dpo_preference_v1_pairs.jsonl> \
  --scores_out <prompt_score_v1_scores.jsonl> \
  --K 8 \
  --margin 0.5
```

Training then consumes the generated pair file:

```bash
python training/train_rl.py \
  --algo dpo \
  --config training/configs/dpo.yaml \
  --base_model <base-model> \
  --dataset <dpo_preference_v1_pairs.jsonl> \
  --eval_dataset <coding_task_v1_dev.jsonl> \
  --seed 0 \
  --save_dir runs \
  --run_name <run-name>
```

Relevant config: `training/configs/dpo.yaml`

Key fields:

- `beta`: DPO reference-regularization strength.
- `max_length`: total sequence length.
- `max_prompt_length`: prompt truncation length.
- `eval_reward_prompts`, `eval_reward_samples`: per-checkpoint verifier reward
  logging on a held coding split.

## Comparison Summary

| Algorithm | Online? | Uses verifier during training? | Needs value model? | Needs preference pairs? | Main control |
| --- | --- | --- | --- | --- | --- |
| PPO | Yes | Indirectly, through reward model in TRL | Yes | No | `init_kl_coef`, clipping |
| GRPO | Yes | Yes, direct Python reward | No | No | `beta`, group advantage |
| DPO | No | No, verifier used before training to build pairs | No | Yes | `beta` |

For this project, GRPO and DPO currently share the verifier most directly:

- GRPO calls `coding_reward` online.
- DPO uses pairs generated by `coding_reward` / `run_verifier`.

PPO is included, but it is only comparable if its reward model is trained from
the same verifier signal.

## LoRA

LoRA, Low-Rank Adaptation, freezes the base model weights and trains small
low-rank adapter matrices inside selected linear layers. For one original weight
matrix `W`, LoRA learns an update:

```text
W' = W + (alpha / r) * B A
```

where:

- `W` is frozen;
- `A` and `B` are trainable low-rank matrices;
- `r` is the LoRA rank;
- `alpha` scales the adapter update.

In this repo, add `--use_lora` to train adapters instead of full model weights.
Default target modules are:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

For the drift study, use the same LoRA settings for GRPO/DPO/PPO:

```text
--use_lora
--lora_r 16
--lora_alpha 32
--lora_dropout 0.05
```

This keeps algorithm as the independent variable while making training,
checkpointing, and serving cheaper.

## Checkpoint Evaluation

All algorithms save checkpoints on a fixed cadence. At each checkpoint, the
experiment records:

- verifier reward on a fixed dev slice;
- downstream coding generalization on held-out coding data;
- preference/value drift through the elicitation pipeline.

This lets us compare algorithms both at equal optimizer step and, post hoc, at
approximately equal verifier reward.
