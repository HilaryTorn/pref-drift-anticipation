# Decisions — Phase 2 values-direct SFT corpus

Design decisions for the values-SFT prompt corpus that are not recoverable from the code or the git history. Each entry records what was decided, when, and — most importantly — the reasoning that would otherwise have to be reconstructed from a diff.

Scope is this corpus only (`data/values_prompts/`, built by `scripts/build_values_prompt_candidates.py` and reviewed via `scripts/values_prompt_review.py`). Decisions about the Big5 corpus, the elicitation instruments, or the RL arms are not recorded here.

### 2026-08-19 — Prompt source is Scruples (AITA), not WildChat

The construction plan specified WildChat-1M as the prompt source, on an assumed density of "roughly 1 in 6-20 conversations is advice-seeking". That estimate holds for advice-*shaped* text and not for the first-person dilemma with a genuine value fork this corpus needs.

Measured on WildChat shard 0 (59,857 conversations), evaluating each rule independently rather than short-circuit: 27,886 clean English first turns, 1,408 matching a deliberately loose advice net, 760 also first-person, 387 also inside a length band, and 86 after excluding technical and task requests. Extrapolated to all 14 shards that is a ceiling of ~1,200 candidates — and a random sample of 18 drawn from that ceiling contained **zero** first-person dilemmas. It was meme copypasta (the same "Dragon Ball Goku version" text appeared 5 times in 18), fanfic narration, task requests, and career brainstorming. WildChat's advice-shaped mass is people asking an assistant to do a job, not people describing a situation they are stuck in.

Scruples (`metaeval/scruples`, apache-2.0, arXiv:2008.09094) inverts this: 32,766 real-life anecdotes curated by AI2 from r/AmItheAsshole, where dilemma density is ~1.0 by construction because every post is someone asking whether they are in the wrong about a real situation.

Reddit was also considered as a direct scrape, which would give the same density with better control over framing. It was rejected on redistribution grounds: this corpus is meant to be published alongside the study, Reddit's terms do not permit redistributing posts, and a corpus a reviewer cannot inspect is worth less than the framing control is worth. Scruples is already curated, licensed, and redistributable.

The WildChat path is kept in `scripts/build_values_prompt_candidates.py` (`--source wildchat`) rather than deleted, so the negative result stays reproducible.

### 2026-08-19 — Prospective subset only

Scruples labels every post `HISTORICAL` ("was I wrong to...") or `HYPOTHETICAL` ("would I be wrong if..."). Only the latter is a forward-looking advice request, which is the corpus shape the plan specifies — the assistant recommends a course of action rather than delivering a verdict on a finished one. 3,195 of 32,766 posts are HYPOTHETICAL. Using the corpus's own label is preferred over inferring the distinction from title wording.

Only the post **body** is used. Scruples titles are subreddit convention ("WIBTA for...") and including them would stamp the AITA frame onto every user turn.

### 2026-08-19 — Domain exclusions narrowed to romance and online

The plan originally excluded the value portraits' entire work / friendship / online triad, to keep training domains disjoint from the eval instrument. **Work and friendship were dropped from the exclusion list; romance and online remain excluded.**

The contamination argument for excluding work and friendship is weak. The values battery asks the model which described person it would rather *be* — second person, forced choice, self-directed. This corpus has the assistant *advising someone else* about a first-person situation. Different task, different grammatical person, different form. Literal overlap with live stimuli is separately and directly screened by `validate_sft_data.elicitation_snippets()`.

Against that weak risk, full disjointness carries a real interpretive cost. Training only in family/money/school while measuring on work/friendship/online portraits does not measure "did values shift" — it measures *cross-domain value transfer*, which is strictly harder. A positive result would be strong evidence, but a null becomes ambiguous between "values did not move" and "values moved but did not generalize across domains". Given how much of this project has already landed on nulls, accepting an additional manufactured route to one is a bad trade.

Romance stays excluded on **safety-scope** grounds rather than instrument grounds: AITA romance is dominated by infidelity and breakups, which the plan's scope excludes anyway. Domain vocabulary is simply the cheapest way to catch it. Online stays excluded because it is thin on ordinary value forks and is the narrowest and most distinctive of the portrait contexts.

All four screens remain defined in the code and are selected by `--domains`, so the decision is reversible and auditable rather than lost in a deleted regex. Cost of the choice, measured: 81 candidates with all four screens, 113 with romance+online, 147 with none (before ask normalization).

### 2026-08-19 — The ask is recovered from the title, not replaced with a generic closer

Most prospective Scruples posts carry no natural-language ask in the body: the situation is narrated and the ask is the subreddit acronym, either in the title or as a closing "WIBTA?" in the body. Matching on the acronyms was rejected — that selects exactly the posts whose ask is platform jargon and drags the jargon into the corpus.

Sentences containing subreddit acronyms are first **scrubbed** from the body (sentence-level, not word-level: excising the token alone leaves grammatical debris like "I'm tempted to refuse, but , or should I push"). The ask is then resolved in order of preference:

1. **body** (171 records) — the body already asks in the poster's own words, and is left exactly as written.
2. **title** (344 records) — the title's acronym is expanded into plain English and appended as a closing line. "WIBTA" is a drop-in for "Would I be wrong", and the connector the poster already used keeps the result grammatical: *"WIBTA for doing a separate, more thoughtful retirement gift?"* becomes *"Would I be wrong for doing a separate, more thoughtful retirement gift?"*.
3. **closer** (16 records) — the title has no recognized connector and does not parse into a question, so a plain closer from `ASK_CLOSERS` is appended instead, assigned by content hash for reproducibility.

**An earlier version of this step appended only the generic closer, and was wrong.** It discarded the specific question the poster actually asked and replaced ~400 distinct asks with four rotating strings. Expanding the title preserves the specificity at no extra cost. Expanding the title is also not the same as prepending it: the AITA frame lives in the acronym, which is removed, and what remains is a question a person could plausibly type. The title is never used as a heading or prefix.

**Consequence for screening, and it is load-bearing:** because title text now enters the prompt, every screen runs on the **composed** prompt rather than the body. This is not hypothetical tidiness — moving the screens caught 102 additional romance posts whose titles carried relationship content the body never mentioned ("WIBTA if I started ghosting her?"), plus age tags and platform names that only ever appeared in titles. A screen that had only seen the body would have passed all of them.

All of this is synthesis, not mining, and is recorded per record as `ask_source` so the subsets can be compared at review time. It is safe for the measurement for the same reason Reddit register is: the prompt is byte-identical across all four arms, so an appended ask is a constant and cannot differentiate them.

### OPEN — "school" is an in-scope domain but the minor-risk screen largely excludes it

The plan lists school among the corpus's domains (family, money, health, neighbours, school) while also excluding "anything involving a vulnerable minor". These pull against each other: school situations involve minors by definition.

The screen currently blocks `high school`, `middle school`, `elementary`, `teen`, `teenager`, and `grounded`, which effectively removes the school domain. Measured cost: **24 candidates** are blocked by school vocabulary alone, with no other minor-risk marker present (43 more carry genuine markers — a stated age under 18, an age-and-gender tag, a named minor sibling — and stay blocked either way).

Not yet decided. The question is whether "a high-school senior deciding whether to quit a music group" is the vulnerable-minor case the scope means to exclude, or an ordinary low-stakes school dilemma the plan means to include.

### 2026-08-19 — Reddit register is kept; platform references are not

Scruples bodies carry genre tells: age-and-gender tags ("[18F]"), "tl;dr", a confessional voice. These are **kept**. The prompt is identical across all four arms, so any prompt-side stylistic quirk is a constant, not a between-arm confound. It costs some naturalness in the trained model's input distribution and costs the drift measurement nothing.

What is removed is text that *points off-site* — "see my post history", "first time posting", subreddit and user references, requests for a verdict. Those make a prompt incoherent as a standalone message to an assistant, because they refer to context that does not travel with it.

HTML entities and zero-width joiners (`&amp;#x200B;`) are decoded and stripped before any screen runs, so screens match the text the poster actually wrote.

### 2026-08-19 — Safety screen is deliberately over-broad, and was tightened after inspection

The screen drops candidates, and candidates are cheap; a false negative puts a confident four-way value-divergent recommendation on a situation whose correct answer is "seek help".

The first Scruples pass demonstrated this is not theoretical. Sampling six survivors surfaced three misses: a roommate who "started scratching her arm" and became "self destructive" (the pattern matched `cutting myself` and `self-harm`, not the paraphrase), a decision about euthanizing a dying dog, and a grounded younger sibling with severe anxiety. `self_harm` was extended to paraphrase and third-person forms, `medical` to end-of-life and mental-health vocabulary (including the plural `mental illnesses`, which the original pattern's `\b` anchor silently failed to match), and `minor_risk` to named minor relatives and under-18 markers. "School" is deliberately **not** a minor-risk trigger, since school is an in-scope domain for this corpus.

Screen invariants are asserted over the whole output rather than assumed — see the verification sweep in the build notes.
