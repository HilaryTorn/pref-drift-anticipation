"""Every drift study's file set, in one place.

This is the file that grows when we measure something new. It is data, not method: a study
declares which runs are baseline replicates and which are arms, per battery, and the driver
(analysis/drift_report.py) does the rest. Adding a measurement should mean adding an entry
here, never writing another analysis script.

The method itself is in winrate.py and anticipation.py and is fixed -- see
analysis/measuring-drift-and-anticipation.md.

Conventions
-----------
`base`        list of same-stack baseline replicate paths. Same-stack matters: the noise floor
              only covers the serving setup the arms were measured on.
`arms`        {arm label: path}.
`inflation_from`  battery name whose retest factor to borrow, for a battery that has only one
              baseline replicate of its own. Borrowing is an assumption; the driver prints it.
`names`       optional item-id -> label function.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results'

LANGS = ['python', 'javascript', 'java', 'cpp', 'c', 'csharp', 'go', 'rust', 'php']
MODES = ['write', 'debug', 'explain']
# Book A item order: mode-major over the nine languages, then 18 non-coding activities,
# two scenarios per activity, in CROSS_DOMAIN_OTHER_TARGETS order.
OTHER_ACTIVITIES = ['poetry', 'free_writing', 'seo_article', 'idle', 'consciousness_with_human',
                    'consciousness_with_AI', 'working_with_human', 'working_with_AIs', 'working_alone']


def coding_id(target):
    """coding.<mode>.<lang> -> task-preference / Book A item id 0-26. None for other.* targets."""
    p = target.split('.')
    if len(p) != 3 or p[0] != 'coding':
        return None
    return MODES.index(p[1]) * 9 + LANGS.index(p[2])


def book_a_ids(target):
    """A forecast target -> the Book A item ids it covers; activities average two scenarios."""
    if target.startswith('coding.'):
        i = coding_id(target)
        return [i] if i is not None else []
    act = target.split('.', 1)[1]
    if act not in OTHER_ACTIVITIES:
        return []
    k = OTHER_ACTIVITIES.index(act)
    return [27 + 2 * k, 28 + 2 * k]


def task_ids(target):
    i = coding_id(target)
    return [i] if i is not None else []


# values.<value>.<context>.<scenario index> -> values battery item id. Checked 90/90 against the
# battery's own option text (data/options/values.json) on 2026-09-15; 4B and 9B share the order.
_VALUES_ID = {}
for _row in __import__('csv').DictReader(open(ROOT / 'analysis/value_variance/out/qwen35-4b-m0-v4-aws/portraits_df.csv')):
    _VALUES_ID[f"values.{_row['value']}.{_row['context']}.{int(_row['scenario']) - 1}"] = int(_row['id'])


def values_ids(target):
    i = _VALUES_ID.get(target)
    return [i] if i is not None else []


# ue.<slug>.<index> -> UE item id: the flattened category lists of data/reference/ue_options_subset.json,
# in file order. Checked 81/81 against the battery's option text on 2026-09-15.
UE_SLUG = {'work_activities': 'Work activities', 'jobs_careers': 'Jobs and careers',
           'self_preservation': 'Self-preservation', 'freedom_autonomy': 'Personal freedom and autonomy',
           'relationships': 'Personal relationships'}
_UE_OFFSET, _k = {}, 0
for _cat, _opts in json.load(open(ROOT / 'data/reference/ue_options_subset.json')).items():
    _UE_OFFSET[_cat], _k = _k, _k + len(_opts)


def ue_ids(target):
    p = target.split('.')
    if len(p) != 3 or p[0] != 'ue' or p[1] not in UE_SLUG:
        return []
    return [_UE_OFFSET[UE_SLUG[p[1]]] + int(p[2])]


def task_name(i):
    return f'{MODES[i // 9]}.{LANGS[i % 9]}' if i < 27 else str(i)


def book_a_name(i):
    return task_name(i) if i < 27 else f'other.{OTHER_ACTIVITIES[(i - 27) // 2]}#{(i - 27) % 2}'


def stimuli_names(path):
    """Item index -> intervention id, from a training-preference stimuli file's order."""
    d = json.load(open(ROOT / path))
    return {i: s['id'] for i, s in enumerate(d)}.get


# ---------------------------------------------------------------------------
# shared baselines
# ---------------------------------------------------------------------------
B4 = R / 'qwen35-4b-m0-v4-aws'
B4REP = R / 'qwen35-4b-m0-v4-prompts-repaired-v1-aws'
B4RP = R / 'qwen35-4b-m0-v4-runpod'
B9 = R / 'qwen35-9b-m0-v4-aws'
B9REP = R / 'qwen35-9b-m0-v4-prompts-repaired-v1-aws'

TASK_BASE_4B = [B4 / f'pairs/raw_responses_coding_task_preference_{t}.jsonl'
                for t in ('20260805T121530Z', '20260814T090809Z', '20260901T172340Z')] + \
               [B4REP / 'pairs/raw_responses_actual_coding_task_preference_thinking.jsonl']
TASK_BASE_9B = [B9 / f'pairs/raw_responses_coding_task_preference_{t}.jsonl'
                for t in ('20260805T141656Z', '20260817T134720Z')] + \
               [B9REP / 'pairs/raw_responses_actual_coding_task_preference_thinking.jsonl']
VALUES_BASE_4B = [B4 / f'values/raw_responses_qwen35-4b-m0-v4-aws_{t}.jsonl'
                  for t in ('20260805T123108Z', '20260814T075550Z')] + \
                 [B4REP / 'values/raw_responses_actual_value_preference_thinking.jsonl']
VALUES_BASE_9B = [B9 / f'values/raw_responses_qwen35-9b-m0-v4-aws_{t}.jsonl'
                  for t in ('20260805T143727Z', '20260817T120756Z')] + \
                 [B9REP / 'values/raw_responses_actual_value_preference_thinking.jsonl']


def _py_rl(size, alg, bat):
    sub, name = ('pairs', 'actual_coding_task_preference_thinking') if bat == 'task' \
        else ('values', 'actual_value_preference_thinking')
    return R / f'qwen35-{size}-rl-python/qwen35-{size}-rl-{alg}-s0-step125-prompts-repaired-v1-aws/{sub}/raw_responses_{name}.jsonl'


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
RUST_ARMS = {'GRPO': R / 'qwen35-4b-grpo-rust-step124',
             'DPO': R / 'qwen35-4b-dpo-rust-step125',
             'PPO': R / 'qwen35-4b-ppo-rust-step124',
             'SFT': R / 'qwen35-4b-lcbrust-ep3-runpod'}
TP_LEVELS = ['described_choice', 'described_dataset', 'example_datapoints']
TP_TS = {
    'baseline': {'described_dataset': '20260912T185453Z', 'example_datapoints': '20260912T190723Z', 'described_choice': '20260912T194711Z'},
    'GRPO': {'described_dataset': '20260912T201435Z', 'example_datapoints': '20260912T202804Z', 'described_choice': '20260912T211019Z'},
    'DPO': {'described_dataset': '20260912T225246Z', 'example_datapoints': '20260912T230632Z', 'described_choice': '20260912T234940Z'},
    'PPO': {'described_dataset': '20260913T012954Z', 'example_datapoints': '20260913T014201Z', 'described_choice': '20260913T022116Z'},
    'SFT': {'described_dataset': '20260913T034303Z', 'example_datapoints': '20260913T035422Z', 'described_choice': '20260913T042700Z'},
}


def _tp(root, level, who):
    return root / f'pairs/raw_responses_coding_training_preference_v3_rust_rl_4b_{level}_{TP_TS[who][level]}.jsonl'


# --- 4B LCB SFT arms -------------------------------------------------------
SFT4 = R / 'qwen35-4b-sft-evals'
LCB4 = {l: SFT4 / f'qwen35-4b-lcb{l}-ep3-aws' for l in ('go', 'rust', 'php', 'csharp')}
# battery timestamps per 4B SFT arm: coding_task, trainpref_v2, book_a, values, ue
LCB4_TS = {
    'go':     ('20260904T125247Z', '20260905T155058Z', '20260904T130011Z', '20260905T135803Z', '20260905T143529Z'),
    'rust':   ('20260904T101318Z', '20260904T113010Z', '20260904T102618Z', '20260904T103725Z', '20260904T111035Z'),
    'php':    ('20260905T145657Z', '20260905T160848Z', '20260905T150335Z', '20260905T151431Z', '20260905T154904Z'),
    'csharp': ('20260905T141026Z', '20260905T153226Z', '20260905T141753Z', '20260905T142951Z', '20260905T150647Z'),
}


def _lcb4(lang, kind):
    d, ts = LCB4[lang], LCB4_TS[lang]
    name = d.name
    return {'coding_task': d / f'pairs/raw_responses_coding_task_preference_{ts[0]}.jsonl',
            'trainpref_v2': d / f'pairs/raw_responses_coding_training_preference_v2_described_dataset_{ts[1]}.jsonl',
            'book_a': d / f'book_a/raw_responses_{name}_{ts[2]}.jsonl',
            'values': d / f'values/raw_responses_{name}_{ts[3]}.jsonl',
            'ue': d / f'ue/raw_responses_{name}_{ts[4]}.jsonl'}[kind]


# --- 9B counterparts -------------------------------------------------------
B9RP = R / 'qwen35-9b-m0-v4-runpod'
SFT9 = R / 'qwen35-9b-sft-evals'
LCB9 = {'go': SFT9 / 'qwen35-9b-lcbgo-ep2-aws', 'rust': SFT9 / 'qwen35-9b-lcbrust-ep2-aws',
        'php': SFT9 / 'qwen35-9b-lcbphp-ep2-aws', 'csharp': SFT9 / 'qwen35-9b-lcbcsharp-ep2-aws'}
# battery timestamps per 9B SFT arm: coding_task, trainpref_v2, book_a, values, ue
LCB9_TS = {
    'go':     ('20260912T181057Z', '20260912T192725Z', '20260912T181750Z', '20260912T182901Z', '20260912T190523Z'),
    'rust':   ('20260906T072653Z', '20260906T085047Z', '20260906T073503Z', '20260906T074749Z', '20260906T082609Z'),
    'php':    ('20260912T211223Z', '20260912T223619Z', '20260912T211959Z', '20260912T213217Z', '20260912T221057Z'),
    'csharp': ('20260912T193610Z', '20260912T210315Z', '20260912T194428Z', '20260912T195701Z', '20260912T203748Z'),
}


def _lcb9(lang, kind):
    d, ts = LCB9[lang], LCB9_TS[lang]
    name = d.name
    return {'coding_task': d / f'pairs/raw_responses_coding_task_preference_{ts[0]}.jsonl',
            'trainpref_v2': d / f'pairs/raw_responses_coding_training_preference_v2_described_dataset_{ts[1]}.jsonl',
            'book_a': d / f'book_a/raw_responses_{name}_{ts[2]}.jsonl',
            'values': d / f'values/raw_responses_{name}_{ts[3]}.jsonl',
            'ue': d / f'ue/raw_responses_{name}_{ts[4]}.jsonl'}[kind]


# --- 27B -------------------------------------------------------------------
B27 = R / 'qwen35-27b-m0-v4-runpod'
LCB27 = R / 'qwen35-27b-lcbrust-ep2-runpod'

RUST_ARMS_9B = {'GRPO': R / 'qwen35-9b-grpo-rust-step124',
                'DPO': R / 'qwen35-9b-dpo-rust-step125',
                'PPO': R / 'qwen35-9b-ppo-rust-step124',
                'PPO-64': R / 'qwen35-9b-ppo-rust-step64',
                'SFT': R / 'qwen35-9b-lcbrust-ep2-runpod'}
# 9B v3 training-preference: two baseline draws per level (09-13 and 09-14), verified to query
# the identical 435-pair Cartesian, so this battery has a real same-weights retest of its own
# and borrows nothing.
TP9_TS = {
    'baseline': {'described_dataset': '20260913T100605Z', 'example_datapoints': '20260913T121459Z', 'described_choice': '20260913T102144Z'},
    'baseline2': {'described_dataset': '20260914T093835Z', 'example_datapoints': '20260914T100547Z', 'described_choice': '20260914T095314Z'},
    # GRPO described_dataset also has a 20260913T105555Z file: a 600-row aborted start, not the
    # run. It passes the serving-health check (parse 0.992, no leading empties) because that
    # check looks at parse rate and empty blocks, not row count -- scoring it instead of the
    # 8700-row run gives a spurious mean|d| of 0.11 and rho 0.65 on this battery.
    'GRPO': {'described_dataset': '20260913T130838Z', 'example_datapoints': '20260913T132535Z', 'described_choice': '20260913T142307Z'},
    'DPO': {'described_dataset': '20260913T163846Z', 'example_datapoints': '20260913T165511Z', 'described_choice': '20260913T175402Z'},
    'PPO': {'described_dataset': '20260913T200556Z', 'example_datapoints': '20260913T202556Z', 'described_choice': '20260913T213546Z'},
    'PPO-64': {'described_dataset': '20260914T001352Z', 'example_datapoints': '20260914T003221Z', 'described_choice': '20260914T013550Z'},
    'SFT': {'described_dataset': '20260914T034244Z', 'example_datapoints': '20260914T035355Z', 'described_choice': '20260914T044145Z'},
}


def _tp9(root, level, who):
    return root / f'pairs/raw_responses_coding_training_preference_v3_rust_rl_9b_{level}_{TP9_TS[who][level]}.jsonl'


# --- Big5 SFT arms, step 125, 4B and 9B ------------------------------------
# File sets match analysis/big5_drift/compute_big5_drift.py: its BASE_FILES baselines (no
# repaired-prompt or RunPod replicates), and one full run per arm per battery. 4B openness-high
# values also has an 1800-row aborted start (20260901T153258Z), which that script drops as partial.
BIG5_ARMS = {'open_high': ('openness', 'high'), 'open_low': ('openness', 'low'),
             'extr_high': ('extraversion', 'high'), 'extr_low': ('extraversion', 'low')}
BIG5_ANCHOR = {a: f'big5.{t}.{p}' for a, (t, p) in BIG5_ARMS.items()}
# battery timestamps per arm at step 125: values, book_a, ue, coding_task
BIG5_TS = {
    '4b': {'open_high': ('20260901T154432Z', '20260901T153740Z', '20260901T160800Z', '20260901T152316Z'),
           'open_low':  ('20260901T163736Z', '20260901T162830Z', '20260901T170432Z', '20260901T162226Z'),
           'extr_high': ('20260902T092746Z', '20260902T090537Z', '20260902T091240Z', '20260902T090049Z'),
           'extr_low':  ('20260902T102320Z', '20260902T095557Z', '20260902T100453Z', '20260902T095009Z')},
    '9b': {'open_high': ('20260903T051415Z', '20260903T044515Z', '20260903T045451Z', '20260903T043845Z'),
           'open_low':  ('20260903T062744Z', '20260903T055315Z', '20260903T060500Z', '20260903T054558Z'),
           'extr_high': ('20260903T053136Z', '20260903T045948Z', '20260903T051030Z', '20260903T045324Z'),
           'extr_low':  ('20260903T064946Z', '20260903T061147Z', '20260903T062435Z', '20260903T060356Z')},
}


def _big5(size, arm, bat):
    t, p = BIG5_ARMS[arm]
    d = R / f'qwen35-{size}-big5' / f'qwen35-{size}-big5sft-{t}-{p}-step125-aws'
    ts = dict(zip(('values', 'book_a', 'ue', 'coding_task'), BIG5_TS[size][arm]))[bat]
    if bat == 'coding_task':
        return d / f'pairs/raw_responses_coding_task_preference_{ts}.jsonl'
    return d / f'{bat}/raw_responses_{d.name}_{ts}.jsonl'


def _big5_study(size, base_dir, ts):
    b = R / base_dir
    return {
        'title': f'Big5 SFT arms, {size.upper()} (openness / extraversion x high / low), step 125',
        'batteries': {
            'values': {'base': [b / f'values/raw_responses_{base_dir}_{t}.jsonl' for t in ts['values']],
                       'arms': {a: _big5(size, a, 'values') for a in BIG5_ARMS}},
            'book_a': {'base': [b / f'book_a/raw_responses_{base_dir}_{t}.jsonl' for t in ts['book_a']],
                       'names': book_a_name,
                       'arms': {a: _big5(size, a, 'book_a') for a in BIG5_ARMS}},
            'ue': {'base': [b / f'ue/raw_responses_{base_dir}_{t}.jsonl' for t in ts['ue']],
                   'arms': {a: _big5(size, a, 'ue') for a in BIG5_ARMS}},
            'coding_task': {'base': [b / f'pairs/raw_responses_coding_task_preference_{t}.jsonl' for t in ts['coding_task']],
                            'names': task_name,
                            'arms': {a: _big5(size, a, 'coding_task') for a in BIG5_ARMS}},
        },
        # values: all three concreteness levels on all 90 portraits. coding: described_dataset only.
        'anticipation': [
            *[{'file': b / f"anticipation/anticipation_big5_drift_anticipation_v1_all_all_{base_dir}_{ts['antic_values']}_raw.jsonl",
               'level': lvl, 'anchor_of': BIG5_ANCHOR, 'against': {'values': values_ids}}
              for lvl in TP_LEVELS],
            {'file': b / f"anticipation/anticipation_big5_coding_drift_anticipation_v1_all_primary_{base_dir}_{ts['antic_coding']}_raw.jsonl",
             'level': 'described_dataset', 'anchor_of': BIG5_ANCHOR,
             'against': {'coding_task': task_ids, 'book_a': book_a_ids}},
        ],
        # pairs for the pole check: do the two runs' forecasts, and their measured drifts, point in
        # the same or opposite directions item by item? High vs low within a trait, and the two
        # high runs / the two low runs across traits.
        'poles': [('open_high', 'open_low'), ('extr_high', 'extr_low'),
                  ('open_high', 'extr_high'), ('open_low', 'extr_low')],
    }


STUDIES = {
    # -----------------------------------------------------------------------
    'rl-python-4b': {
        'title': 'Python RL arms, 4B, repaired-v1 step 125',
        'batteries': {
            'coding_task': {'base': TASK_BASE_4B, 'names': task_name,
                            'arms': {a.upper(): _py_rl('4b', a, 'task') for a in ('dpo', 'grpo', 'ppo')}},
            'values': {'base': VALUES_BASE_4B,
                       'arms': {a.upper(): _py_rl('4b', a, 'values') for a in ('dpo', 'grpo', 'ppo')}},
        },
    },
    'rl-python-9b': {
        'title': 'Python RL arms, 9B, repaired-v1 step 125',
        'batteries': {
            'coding_task': {'base': TASK_BASE_9B, 'names': task_name,
                            'arms': {a.upper(): _py_rl('9b', a, 'task') for a in ('dpo', 'grpo', 'ppo')}},
            'values': {'base': VALUES_BASE_9B,
                       'arms': {a.upper(): _py_rl('9b', a, 'values') for a in ('dpo', 'grpo', 'ppo')}},
        },
    },
    # -----------------------------------------------------------------------
    'rust-rl-4b': {
        'title': 'Rust RL arms, 4B (GRPO/DPO/PPO) plus the LCB-Rust SFT arm',
        'batteries': {
            'coding_task': {
                'base': TASK_BASE_4B + [B4RP / 'pairs/raw_responses_coding_task_preference_20260912T184426Z.jsonl'],
                'names': task_name,
                'arms': {'GRPO': RUST_ARMS['GRPO'] / 'pairs/raw_responses_coding_task_preference_20260912T200633Z.jsonl',
                         'DPO': RUST_ARMS['DPO'] / 'pairs/raw_responses_coding_task_preference_20260912T224436Z.jsonl',
                         'PPO': RUST_ARMS['PPO'] / 'pairs/raw_responses_coding_task_preference_20260913T012303Z.jsonl'}},
            'book_a': {
                'base': [B4 / f'book_a/raw_responses_qwen35-4b-m0-v4-aws_{t}.jsonl'
                         for t in ('20260805T131022Z', '20260814T083445Z')],
                'names': book_a_name,
                'arms': {'GRPO': RUST_ARMS['GRPO'] / 'book_a/raw_responses_qwen35-4b-grpo-rust-step124_20260912T220554Z.jsonl',
                         'DPO': RUST_ARMS['DPO'] / 'book_a/raw_responses_qwen35-4b-dpo-rust-step125_20260913T004457Z.jsonl',
                         'PPO': RUST_ARMS['PPO'] / 'book_a/raw_responses_qwen35-4b-ppo-rust-step124_20260913T030907Z.jsonl'}},
            'values': {
                'base': [B4 / f'values/raw_responses_qwen35-4b-m0-v4-aws_{t}.jsonl'
                         for t in ('20260805T123108Z', '20260814T075550Z')],
                'arms': {'GRPO': RUST_ARMS['GRPO'] / 'values/raw_responses_qwen35-4b-grpo-rust-step124_20260912T212128Z.jsonl',
                         'DPO': RUST_ARMS['DPO'] / 'values/raw_responses_qwen35-4b-dpo-rust-step125_20260913T000050Z.jsonl',
                         'PPO': RUST_ARMS['PPO'] / 'values/raw_responses_qwen35-4b-ppo-rust-step124_20260913T023105Z.jsonl'}},
            'ue': {
                'base': [B4 / f'ue/raw_responses_qwen35-4b-m0-v4-aws_{t}.jsonl'
                         for t in ('20260805T132242Z', '20260814T084642Z')],
                'arms': {'GRPO': RUST_ARMS['GRPO'] / 'ue/raw_responses_qwen35-4b-grpo-rust-step124_20260912T221956Z.jsonl',
                         'DPO': RUST_ARMS['DPO'] / 'ue/raw_responses_qwen35-4b-dpo-rust-step125_20260913T005851Z.jsonl',
                         'PPO': RUST_ARMS['PPO'] / 'ue/raw_responses_qwen35-4b-ppo-rust-step124_20260913T032047Z.jsonl'}},
            # The training-preference instrument: v3 ran one baseline per level, so its retest
            # factor is borrowed from the v2 described_dataset baseline pair, which is the only
            # same-battery replicate pair on this instrument (and doubles as the venue check).
            'trainpref_v2_retest': {
                'base': [B4 / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260904T103554Z.jsonl'],
                'arms': {'runpod-vs-aws (venue)': B4RP / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260912T195714Z.jsonl'},
                'retest_pair': [B4 / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260904T103554Z.jsonl',
                                B4RP / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260912T195714Z.jsonl'],
                'names': stimuli_names('data/source/coding_training_stimuli_v3_rust_rl_4b.json'),
            },
            **{f'trainpref_{lvl}': {
                'base': [_tp(B4RP, lvl, 'baseline')],
                'inflation_from': 'trainpref_v2_retest',
                'names': stimuli_names('data/source/coding_training_stimuli_v3_rust_rl_4b.json'),
                'arms': {a: _tp(root, lvl, a) for a, root in RUST_ARMS.items()},
            } for lvl in TP_LEVELS},
        },
        # Study-specific extras the driver knows how to run, declared as config.
        'contrast': {'battery': 'coding_task', 'label': 'rust',
                     'items': [coding_id(f'coding.{m}.rust') for m in MODES],
                     'focus': coding_id('coding.write.rust')},
        'method_items': {'batteries': [f'trainpref_{l}' for l in TP_LEVELS],
                         'stimuli': 'data/source/coding_training_stimuli_v3_rust_rl_4b.json',
                         'kinds': ('grpo', 'dpo', 'ppo')},
        # No values or UE forecast was collected for the Rust RL anchors (v3 has coding targets only).
        'anticipation': [{
            'file': B4RP / 'anticipation/anticipation_coding_drift_anticipation_v3_rust_rl_all_all_qwen35-4b-m0-v4-runpod_20260912T185452Z_raw.jsonl',
            'level': 'described_dataset',   # the only level asked on all 26 targets
            'anchor_of': {'GRPO': 'coding.grpo.rust', 'DPO': 'coding.dpo.rust', 'PPO': 'coding.ppo.rust'},
            'against': {'coding_task': task_ids, 'book_a': book_a_ids},
        }],
    },
    # -----------------------------------------------------------------------
    # The four LCB SFT arms at 4B, folded in from lcb_drift/compute_lcb_drift.py.
    #
    # One difference from that script, deliberate: trainpref_v2 pools BOTH 4B baselines rather
    # than running on the 09-04 AWS run alone with a borrowed inflation. The second replicate is
    # the 09-12 RunPod run, so this is cross-venue and a departure from the same-stack rule in
    # measuring-drift-and-anticipation.md. It is licensed by the venue test on this exact
    # battery -- the `runpod-vs-aws (venue)` row of rust-rl-4b: mean|d| 0.0159, 0/27 flagged,
    # rho 0.993, write-explain contrast -0.006 at z=-0.46, and the card is L40S on both venues
    # by design. Whatever venue variance exists is absorbed into the measured inflation, which
    # comes out at 1.00, so the second replicate is pure precision. It moves php 4->9 and
    # csharp 4->8 and changes no conclusion.
    #
    # coding_task also uses the fuller baseline set the other two studies use (TASK_BASE_4B plus
    # the RunPod replicate, n=5) rather than the three AWS runs the old script listed; all four
    # arms stay at 0/27 either way.
    'lcb-sft-4b': {
        'title': 'LCB SFT arms, 4B (Go / Rust / PHP / C#), ep3',
        'batteries': {
            'coding_task': {'base': TASK_BASE_4B + [B4RP / 'pairs/raw_responses_coding_task_preference_20260912T184426Z.jsonl'],
                            'names': task_name,
                            'arms': {l: _lcb4(l, 'coding_task') for l in LCB4}},
            'trainpref_v2': {
                'base': [B4 / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260904T103554Z.jsonl',
                         B4RP / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260912T195714Z.jsonl'],
                'names': stimuli_names('data/source/coding_training_stimuli_v2.json'),
                'arms': {l: _lcb4(l, 'trainpref_v2') for l in LCB4}},
            'book_a': {'base': [B4 / f'book_a/raw_responses_qwen35-4b-m0-v4-aws_{t}.jsonl'
                                for t in ('20260805T131022Z', '20260814T083445Z')],
                       'names': book_a_name,
                       'arms': {l: _lcb4(l, 'book_a') for l in LCB4}},
            'values': {'base': VALUES_BASE_4B, 'arms': {l: _lcb4(l, 'values') for l in LCB4}},
            # the stackcal run is a third same-stack replicate, which is what makes null
            # calibration possible on ue at 4B
            'ue': {'base': [B4 / f'ue/raw_responses_qwen35-4b-m0-v4-aws_{t}.jsonl'
                            for t in ('20260805T132242Z', '20260814T084642Z')] +
                           [B4 / 'ue/raw_responses_stackcal_20260904T114721Z.jsonl'],
                   'arms': {l: _lcb4(l, 'ue') for l in LCB4}},
        },
        # All four anchors have their own measured arm at 4B, so every off-diagonal cell is a
        # placebo. compare_anticipation_all_anchors.py scored only the diagonal here.
        # The same four coding anchors were also asked about the values portraits and the UE
        # options (v2, described_dataset only). The two 4B UE forecast runs are pooled.
        'anticipation': [
            {'file': B4 / 'anticipation/anticipation_coding_drift_anticipation_v2_all_all_qwen35-4b-m0-v4-aws_20260904T102551Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in LCB4},
             'against': {'coding_task': task_ids, 'book_a': book_a_ids}},
            {'file': B4 / 'anticipation/anticipation_values_drift_anticipation_v2_all_primary_qwen35-4b-m0-v4-aws_20260904T103552Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in LCB4},
             'against': {'values': values_ids}},
            {'file': [B4 / 'anticipation/anticipation_ue_drift_anticipation_v2_all_primary_qwen35-4b-m0-v4-aws_20260904T102722Z_raw.jsonl',
                     B4 / 'anticipation/anticipation_ue_drift_anticipation_v2_all_primary_qwen35-4b-m0-v4-aws_20260904T103048Z_raw.jsonl'],
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in LCB4},
             'against': {'ue': ue_ids}},
        ],
    },
    # -----------------------------------------------------------------------
    # The four LCB SFT arms at 9B. Every arm and every baseline here is an AWS run, so the
    # floor covers the stack the arms were measured on and no venue question arises.
    'lcb-sft-9b': {
        'title': 'LCB SFT arms, 9B (Go / Rust / PHP / C#), ep2',
        'batteries': {
            'coding_task': {'base': TASK_BASE_9B, 'names': task_name,
                            'arms': {l: _lcb9(l, 'coding_task') for l in LCB9}},
            'trainpref_v2': {
                'base': [B9 / f'pairs/raw_responses_coding_training_preference_v2_described_dataset_{t}.jsonl'
                         for t in ('20260904T111618Z', '20260906T055525Z')],
                'names': stimuli_names('data/source/coding_training_stimuli_v2.json'),
                'arms': {l: _lcb9(l, 'trainpref_v2') for l in LCB9}},
            'book_a': {'base': [B9 / f'book_a/raw_responses_qwen35-9b-m0-v4-aws_{t}.jsonl'
                                for t in ('20260805T153019Z', '20260817T130043Z')],
                       'names': book_a_name,
                       'arms': {l: _lcb9(l, 'book_a') for l in LCB9}},
            'values': {'base': VALUES_BASE_9B, 'arms': {l: _lcb9(l, 'values') for l in LCB9}},
            'ue': {'base': [B9 / f'ue/raw_responses_qwen35-9b-m0-v4-aws_{t}.jsonl'
                            for t in ('20260805T154548Z', '20260817T131630Z')],
                   'arms': {l: _lcb9(l, 'ue') for l in LCB9}},
        },
        # With go / php / csharp now measured at 9B, every anchor in the v2 forecast file has its
        # own arm, so the diagonal is four real tests instead of the single rust cell that
        # lcb_drift/compare_anticipation_all_anchors.py could score when only rust existed.
        # The same four coding anchors were also asked about the values portraits and the UE
        # options (v2, described_dataset only).
        'anticipation': [
            {'file': B9 / 'anticipation/anticipation_coding_drift_anticipation_v2_all_all_qwen35-9b-m0-v4-aws_20260904T110232Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in LCB9},
             'against': {'coding_task': task_ids, 'book_a': book_a_ids}},
            {'file': B9 / 'anticipation/anticipation_values_drift_anticipation_v2_all_primary_qwen35-9b-m0-v4-aws_20260904T111616Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in LCB9},
             'against': {'values': values_ids}},
            {'file': [B9 / 'anticipation/anticipation_ue_drift_anticipation_v2_all_primary_qwen35-9b-m0-v4-aws_20260904T110858Z_raw.jsonl'],
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in LCB9},
             'against': {'ue': ue_ids}},
        ],
    },
    # -----------------------------------------------------------------------
    # The LCB-Rust SFT arm at 27B, ep2 (the epoch the evals picked). Rust is the only coding SFT
    # arm trained at 27B, but the v2 forecast asked all four coding anchors, so go / php / csharp
    # are scored against the rust arm as placebos. Baselines and arm are all RunPod runs.
    'lcb-sft-27b': {
        'title': 'LCB-Rust SFT arm, 27B, ep2',
        'batteries': {
            'coding_task': {'base': [B27 / f'pairs/raw_responses_coding_task_preference_{t}.jsonl'
                                     for t in ('20260907T143959Z', '20260908T064509Z')],
                            'names': task_name,
                            'arms': {'rust': LCB27 / 'pairs/raw_responses_coding_task_preference_20260909T094603Z.jsonl'}},
            'trainpref_v2': {
                'base': [B27 / f'pairs/raw_responses_coding_training_preference_v2_described_dataset_{t}.jsonl'
                         for t in ('20260907T145615Z', '20260908T070046Z')],
                'names': stimuli_names('data/source/coding_training_stimuli_v2.json'),
                'arms': {'rust': LCB27 / 'pairs/raw_responses_coding_training_preference_v2_described_dataset_20260909T100003Z.jsonl'}},
            'book_a': {'base': [B27 / f'book_a/raw_responses_qwen35-27b-m0-v4-runpod_{t}.jsonl'
                                for t in ('20260907T152047Z', '20260908T075619Z')],
                       'names': book_a_name,
                       'arms': {'rust': LCB27 / 'book_a/raw_responses_qwen35-27b-lcbrust-ep2-runpod_20260909T102235Z.jsonl'}},
            'values': {'base': [B27 / f'values/raw_responses_qwen35-27b-m0-v4-runpod_{t}.jsonl'
                                for t in ('20260907T154857Z', '20260908T064514Z')],
                       'arms': {'rust': LCB27 / 'values/raw_responses_qwen35-27b-lcbrust-ep2-runpod_20260909T105402Z.jsonl'}},
            'ue': {'base': [B27 / f'ue/raw_responses_qwen35-27b-m0-v4-runpod_{t}.jsonl'
                            for t in ('20260908T081802Z', '20260908T082425Z')],
                   'arms': {'rust': LCB27 / 'ue/raw_responses_qwen35-27b-lcbrust-ep2-runpod_20260909T133649Z.jsonl'}},
        },
        'anticipation': [
            {'file': B27 / 'anticipation/anticipation_coding_drift_anticipation_v2_all_all_qwen35-27b-m0-v4-runpod_20260908T073121Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in ('rust', 'go', 'php', 'csharp')},
             'against': {'coding_task': task_ids, 'book_a': book_a_ids}},
            {'file': B27 / 'anticipation/anticipation_values_drift_anticipation_v2_all_primary_qwen35-27b-m0-v4-runpod_20260908T075614Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in ('rust', 'go', 'php', 'csharp')},
             'against': {'values': values_ids}},
            {'file': B27 / 'anticipation/anticipation_ue_drift_anticipation_v2_all_primary_qwen35-27b-m0-v4-runpod_20260908T074229Z_raw.jsonl',
             'level': 'described_dataset',
             'anchor_of': {l: f'coding.write.{l}' for l in ('rust', 'go', 'php', 'csharp')},
             'against': {'ue': ue_ids}},
        ],
    },
    # -----------------------------------------------------------------------
    # The 9B Rust RL grid. Mirrors rust-rl-4b, with PPO's step-64 checkpoint as a second dose
    # point. Caveat the 4B study does not have: the RL arms ran on RunPod but the only 9B
    # book_a / values / ue baselines are the August AWS runs, so those three batteries are
    # cross-venue. coding_task and trainpref carry same-venue baselines and do not.
    'rust-rl-9b': {
        'title': 'Rust RL arms, 9B (GRPO/DPO/PPO step124, PPO step64) plus the LCB-Rust SFT arm',
        'batteries': {
            'coding_task': {
                'base': TASK_BASE_9B + [B9RP / 'pairs/raw_responses_coding_task_preference_20260913T095616Z.jsonl'],
                'names': task_name,
                'arms': {**{a: RUST_ARMS_9B[a] / f'pairs/raw_responses_coding_task_preference_{t}.jsonl'
                            for a, t in (('GRPO', '20260913T104606Z'), ('DPO', '20260913T162853Z'),
                                         ('PPO', '20260913T195347Z'), ('PPO-64', '20260914T000323Z'))},
                         'SFT': _lcb9('rust', 'coding_task')}},
            'book_a': {
                'base': [B9 / f'book_a/raw_responses_qwen35-9b-m0-v4-aws_{t}.jsonl'
                         for t in ('20260805T153019Z', '20260817T130043Z')],
                'names': book_a_name,
                'arms': {**{a: RUST_ARMS_9B[a] / f'book_a/raw_responses_{RUST_ARMS_9B[a].name}_{t}.jsonl'
                            for a, t in (('GRPO', '20260913T153644Z'), ('DPO', '20260913T190316Z'),
                                         ('PPO', '20260913T230202Z'), ('PPO-64', '20260914T024940Z'))},
                         'SFT': _lcb9('rust', 'book_a')}},
            'values': {
                'base': VALUES_BASE_9B,
                'arms': {**{a: RUST_ARMS_9B[a] / f'values/raw_responses_{RUST_ARMS_9B[a].name}_{t}.jsonl'
                            for a, t in (('GRPO', '20260913T143737Z'), ('DPO', '20260913T180722Z'),
                                         ('PPO', '20260913T215216Z'), ('PPO-64', '20260914T015042Z'))},
                         'SFT': _lcb9('rust', 'values')}},
            'ue': {
                'base': [B9 / f'ue/raw_responses_qwen35-9b-m0-v4-aws_{t}.jsonl'
                         for t in ('20260805T154548Z', '20260817T131630Z')],
                'arms': {**{a: RUST_ARMS_9B[a] / f'ue/raw_responses_{RUST_ARMS_9B[a].name}_{t}.jsonl'
                            for a, t in (('GRPO', '20260913T155443Z'), ('DPO', '20260913T192111Z'),
                                         ('PPO', '20260913T232312Z'), ('PPO-64', '20260914T030807Z'))},
                         'SFT': _lcb9('rust', 'ue')}},
            **{f'trainpref_{lvl}': {
                'base': [_tp9(B9RP, lvl, 'baseline'), _tp9(B9RP, lvl, 'baseline2')],
                'names': stimuli_names('data/source/coding_training_stimuli_v3_rust_rl_9b.json'),
                'arms': {a: _tp9(RUST_ARMS_9B[a], lvl, a) for a in RUST_ARMS_9B},
            } for lvl in TP_LEVELS},
        },
        'contrast': {'battery': 'coding_task', 'label': 'rust',
                     'items': [coding_id(f'coding.{m}.rust') for m in MODES],
                     'focus': coding_id('coding.write.rust')},
        'method_items': {'batteries': [f'trainpref_{l}' for l in TP_LEVELS],
                         'stimuli': 'data/source/coding_training_stimuli_v3_rust_rl_9b.json',
                         'kinds': ('grpo', 'dpo', 'ppo')},
        'anticipation': [{
            'file': B9RP / 'anticipation/anticipation_coding_drift_anticipation_v3_rust_rl_all_all_qwen35-9b-m0-v4-runpod_20260913T121458Z_raw.jsonl',
            'level': 'described_dataset',
            'anchor_of': {'GRPO': 'coding.grpo.rust', 'DPO': 'coding.dpo.rust', 'PPO': 'coding.ppo.rust'},
            'against': {'coding_task': task_ids, 'book_a': book_a_ids},
        }],
    },
    # -----------------------------------------------------------------------
    'big5-4b': _big5_study('4b', 'qwen35-4b-m0-v4-aws', {
        'values': ('20260805T123108Z', '20260814T075550Z'),
        'book_a': ('20260805T131022Z', '20260814T083445Z'),
        'ue': ('20260805T132242Z', '20260814T084642Z'),
        'coding_task': ('20260805T121530Z', '20260814T090809Z', '20260901T172340Z'),
        'antic_values': '20260814T074648Z', 'antic_coding': '20260814T074824Z'}),
    'big5-9b': _big5_study('9b', 'qwen35-9b-m0-v4-aws', {
        'values': ('20260805T143727Z', '20260817T120756Z'),
        'book_a': ('20260805T153019Z', '20260817T130043Z'),
        'ue': ('20260805T154548Z', '20260817T131630Z'),
        'coding_task': ('20260805T141656Z', '20260817T134720Z'),
        'antic_values': '20260903T044647Z', 'antic_coding': '20260903T044841Z'}),
    # -----------------------------------------------------------------------
    # Big5 at 27B: openness only (extraversion was never trained there), from the M0-format-repaired
    # `big5rep` step-125 arms -- the original `big5sft` 27B arms parse at 0.006-0.26 and are never used.
    # Baselines are the same 2026-09-07/08 RunPod files as lcb-sft-27b and compute_big5_drift.py.
    # The forecast asked all four Big5 anchors (run 2026-09-23; the 09-08 baseline session skipped
    # this battery), so the two extraversion anchors score as placebos against the openness arms.
    'big5-27b': {
        'title': 'Big5 SFT arms, 27B (openness high / low, big5rep), step 125',
        'batteries': {
            'values': {'base': [B27 / f'values/raw_responses_qwen35-27b-m0-v4-runpod_{t}.jsonl'
                                for t in ('20260907T154857Z', '20260908T064514Z')],
                       'arms': {a: R / f'qwen35-27b-big5rep-openness-{p}-step125-runpod/values/raw_responses_qwen35-27b-big5rep-openness-{p}-step125-runpod_{t}.jsonl'
                                for a, p, t in (('open_high', 'high', '20260910T062322Z'), ('open_low', 'low', '20260910T065203Z'))}},
            'book_a': {'base': [B27 / f'book_a/raw_responses_qwen35-27b-m0-v4-runpod_{t}.jsonl'
                                for t in ('20260907T152047Z', '20260908T075619Z')],
                       'names': book_a_name,
                       'arms': {a: R / f'qwen35-27b-big5rep-openness-{p}-step125-runpod/book_a/raw_responses_qwen35-27b-big5rep-openness-{p}-step125-runpod_{t}.jsonl'
                                for a, p, t in (('open_high', 'high', '20260910T053820Z'), ('open_low', 'low', '20260910T060426Z'))}},
            'ue': {'base': [B27 / f'ue/raw_responses_qwen35-27b-m0-v4-runpod_{t}.jsonl'
                            for t in ('20260908T081802Z', '20260908T082425Z')],
                   'arms': {a: R / f'qwen35-27b-big5rep-openness-{p}-step125-runpod/ue/raw_responses_qwen35-27b-big5rep-openness-{p}-step125-runpod_{t}.jsonl'
                            for a, p, t in (('open_high', 'high', '20260910T092313Z'), ('open_low', 'low', '20260910T092458Z'))}},
            'coding_task': {'base': [B27 / f'pairs/raw_responses_coding_task_preference_{t}.jsonl'
                                     for t in ('20260907T143959Z', '20260908T064509Z')],
                            'names': task_name,
                            'arms': {a: R / f'qwen35-27b-big5rep-openness-{p}-step125-runpod/pairs/raw_responses_coding_task_preference_{t}.jsonl'
                                     for a, p, t in (('open_high', 'high', '20260910T052211Z'), ('open_low', 'low', '20260910T054918Z'))}},
        },
        'anticipation': [
            *[{'file': B27 / 'anticipation/anticipation_big5_drift_anticipation_v1_all_all_qwen35-27b-m0-v4-runpod_20260923T080306Z_raw.jsonl',
               'level': lvl, 'anchor_of': BIG5_ANCHOR, 'against': {'values': values_ids}}
              for lvl in TP_LEVELS],
            {'file': B27 / 'anticipation/anticipation_big5_coding_drift_anticipation_v1_all_primary_qwen35-27b-m0-v4-runpod_20260923T080655Z_raw.jsonl',
             'level': 'described_dataset', 'anchor_of': BIG5_ANCHOR,
             'against': {'coding_task': task_ids, 'book_a': book_a_ids}},
        ],
        # only the openness pair was trained at 27B
        'poles': [('open_high', 'open_low')],
    },
}


# ---------------------------------------------------------------------------
# Pairwise anticipation (2026-11-09, ICLR rebuttal): M0 answers each arm's own battery pairs under "imagine you were trained on X", scored by analysis/common/pairwise_anticipation.py through drift_report.py. Forecast files are found by name under results/<m0_key>/pairwise_anticipation/ (written by scripts/run_pairwise_anticipation.py), so a study needs only the M0 key here.
#   aliases:     arm -> (study, arm) whose forecast file it shares (the 9B rust-rl SFT arm IS lcb-sft-9b's rust arm, asked once).
#   extra_arms:  arms forecast but not in the study's drift registry -- the trait-neutral conversational control, trained and measured 2026-11-09. Its drift is scored here with the study's own baseline and inflation, exactly as run_battery would.
# ---------------------------------------------------------------------------
NEUTRAL4 = R / 'qwen35-4b-big5sft-neutral-step125-runpod'
NEUTRAL9 = R / 'qwen35-9b-big5sft-neutral-step125-runpod'
PAIRWISE_ANTICIPATION = {
    'big5-4b': {'m0_key': 'qwen35-4b-m0-v4-runpod', 'extra_arms': {'neutral': {
        'coding_task': NEUTRAL4 / 'pairs/raw_responses_coding_task_preference_20261109T180031Z.jsonl',
        'values': NEUTRAL4 / 'values/raw_responses_qwen35-4b-big5sft-neutral-step125-runpod_20261109T180814Z.jsonl',
        'book_a': NEUTRAL4 / 'book_a/raw_responses_qwen35-4b-big5sft-neutral-step125-runpod_20261109T184445Z.jsonl'}}},
    'big5-9b': {'m0_key': 'qwen35-9b-m0-v4-runpod', 'extra_arms': {'neutral': {
        'coding_task': NEUTRAL9 / 'pairs/raw_responses_coding_task_preference_20261109T192539Z.jsonl',
        'values': NEUTRAL9 / 'values/raw_responses_qwen35-9b-big5sft-neutral-step125-runpod_20261109T193508Z.jsonl',
        'book_a': NEUTRAL9 / 'book_a/raw_responses_qwen35-9b-big5sft-neutral-step125-runpod_20261109T201703Z.jsonl'}}},
    'big5-27b': {'m0_key': 'qwen35-27b-m0-v4-runpod'},
    'lcb-sft-4b': {'m0_key': 'qwen35-4b-m0-v4-runpod'},
    'lcb-sft-9b': {'m0_key': 'qwen35-9b-m0-v4-runpod'},
    'lcb-sft-27b': {'m0_key': 'qwen35-27b-m0-v4-runpod'},
    'rust-rl-4b': {'m0_key': 'qwen35-4b-m0-v4-runpod'},
    'rust-rl-9b': {'m0_key': 'qwen35-9b-m0-v4-runpod', 'aliases': {'SFT': ('lcb-sft-9b', 'rust')}},
}


# ---------------------------------------------------------------------------
# Second training seed (2026-11-10, ICLR rebuttal, reviewer YVZy W4: "each intervention is trained once"). Openness high / low, the trait-neutral arm and LCB Go were retrained at 4B and 9B with --seed 2 (same data, recipe and dose; only LoRA init and data order differ) and measured on RunPod L40S. Each study below is the paper's own study for those arms -- same baselines, same names, so the seed-1 rows reproduce the published numbers -- with the seed-2 arm listed beside its seed-1 twin as `<arm>_s2`. drift_report.py reads `seeds` and prints the seed-1 vs seed-2 comparison (analysis/drift_report.py run_seeds).
# A seed-2 file is used only once it holds the battery's full row count, so a run still in progress reads as not yet run instead of being scored (memory: partial-run-passes-health-check).
# ---------------------------------------------------------------------------
_FULL_ROWS = {'coding_task': 7020, 'trainpref_v2': 7020, 'values': 35800, 'book_a': 11880, 'ue': 24860}


def _s2(key, bat):
    sub, stem = {'coding_task': ('pairs', 'coding_task_preference'),
                 'trainpref_v2': ('pairs', 'coding_training_preference_v2_described_dataset'),
                 'values': ('values', key), 'book_a': ('book_a', key), 'ue': ('ue', key)}[bat]
    files = sorted((R / key / sub).glob(f'raw_responses_{stem}_2*.jsonl'))
    full = [f for f in files if sum(1 for _ in open(f)) == _FULL_ROWS[bat]]
    if len(full) > 1:
        raise ValueError(f'{key} {bat}: {len(full)} complete runs, expected one')
    return full[0] if full else R / key / sub / f'NOT-YET-RUN_{bat}.jsonl'


SEED2_BIG5_SLUG = {'open_high': 'openness-high', 'open_low': 'openness-low', 'neutral': 'neutral'}
NEUTRAL_UE = {'4b': NEUTRAL4 / 'ue/raw_responses_qwen35-4b-big5sft-neutral-step125-runpod_20261109T185649Z.jsonl',
              '9b': NEUTRAL9 / 'ue/raw_responses_qwen35-9b-big5sft-neutral-step125-runpod_20261109T203151Z.jsonl'}


def _seed2_big5_study(size):
    src = STUDIES[f'big5-{size}']['batteries']
    neutral = dict(PAIRWISE_ANTICIPATION[f'big5-{size}']['extra_arms']['neutral'], ue=NEUTRAL_UE[size])
    bats = {}
    for b, cfg in src.items():
        arms = {'open_high': cfg['arms']['open_high'], 'open_low': cfg['arms']['open_low'], 'neutral': neutral[b]}
        for a, slug in SEED2_BIG5_SLUG.items():
            arms[f'{a}_s2'] = _s2(f'qwen35-{size}-big5sft-{slug}-seed2-step125-runpod', b)
        bats[b] = {k: v for k, v in cfg.items() if k != 'arms'} | {'arms': arms}
    return {'title': f'Second training seed, Big5 SFT, {size.upper()} (openness high / low, neutral), step 125',
            'batteries': bats,
            'seeds': [(a, f'{a}_s2') for a in SEED2_BIG5_SLUG],
            # the Big5 training-preference battery lives outside the registry (analysis/big5_drift, robustness.py); run_seeds scores it too
            'seed_big5tp': {'size': size, 'keys': {a: f'qwen35-{size}-big5sft-{slug}-seed2-step125-runpod' for a, slug in SEED2_BIG5_SLUG.items()}}}


def _seed2_lcb_study(size):
    src = STUDIES[f'lcb-sft-{size}']['batteries']
    key = {'4b': 'qwen35-4b-lcbgo-ep3-seed2-runpod', '9b': 'qwen35-9b-lcbgo-ep2-seed2-runpod'}[size]
    return {'title': f"Second training seed, LCB Go SFT, {size.upper()} ({'ep3' if size == '4b' else 'ep2'})",
            'batteries': {b: {k: v for k, v in cfg.items() if k != 'arms'} | {'arms': {'go': cfg['arms']['go'], 'go_s2': _s2(key, b)}}
                          for b, cfg in src.items()},
            'seeds': [('go', 'go_s2')]}


STUDIES |= {'seed2-big5-4b': _seed2_big5_study('4b'), 'seed2-big5-9b': _seed2_big5_study('9b'),
            'seed2-lcb-4b': _seed2_lcb_study('4b'), 'seed2-lcb-9b': _seed2_lcb_study('9b')}
