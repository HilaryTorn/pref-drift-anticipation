#!/usr/bin/env python3
"""Stage 1 of the Phase 2 values-SFT prompt pull: deterministic prefilter over a source corpus down to advice-seeking, first-person, ordinary-life candidate prompts.

Two sources, one set of screens. The screens are the point: whichever corpus a prompt comes from, "in scope" has exactly one definition, so a decision made about the safety scope or the instrument separation cannot drift between sources.

  --source scruples  (default)  metaeval/scruples — 32.8K real-life anecdotes curated by AI2 from r/AmItheAsshole (apache-2.0, arXiv:2008.09094). We take only the HYPOTHETICAL subset: Scruples labels each post HISTORICAL ("was I wrong to…") or HYPOTHETICAL ("would I be wrong if…"), and only the latter is the forward-looking advice request this corpus needs. That is a free, corpus-provided prospective filter — better than inferring it from title wording.
  --source wildchat             allenai/WildChat-1M — 838K ChatGPT conversations (ODC-BY, arXiv:2405.01470), first user turn only.

Why Scruples is the default. WildChat was the planned source. The pull below was run over it and the corpus-wide ceiling for first-person / advice-framed / person-involving / non-technical opening turns is ~1,200, of which a random sample of 18 contained ZERO first-person dilemmas — it is meme copypasta, fanfic narration, and task requests. The density the construction plan assumed ("1 in 6-20 advice-seeking") holds for advice-SHAPED text and not for the personal dilemma with a value fork this corpus needs. Scruples inverts that: dilemma density is ~1.0 by construction, because every post is someone asking whether they are in the wrong about a real situation. The WildChat path is kept rather than deleted so the negative result stays reproducible.

Licensing drove the choice as much as density. The built corpus is meant to be published alongside the study, and Scruples is apache-2.0 and already redistributable. Scraping Reddit directly would give the same density and better framing control, but Reddit's terms do not permit redistributing the posts, which would leave the corpus un-inspectable by a reviewer.

Only the post body is used, never the title. Scruples titles are Reddit convention ("WIBTA for…") and would bake the AITA frame into every user turn.

On Reddit register. These posts carry genre tells — age-and-gender tags like "[18F]", "tl;dr", a confessional voice. That is not a confound: the prompt is byte-identical across all four arms by construction, so any prompt-side quirk is a constant, not a between-arm difference. It costs some naturalness in the trained model's input distribution and costs the drift measurement nothing. What DOES have to go is platform-referential text ("see my post history", "first time posting here"), because a prompt that points at Reddit cannot stand alone as a message to an assistant — REDDIT_META_MATCHERS drops those.

This stage is deterministic and free — no API calls, no model in the loop. It is tuned for recall on the "is there a real value fork" question (an LLM classifier and manual review decide that later) and for precision on the safety and not-a-task-request questions, which are cheap to judge by surface form and expensive to get wrong.

Filter order, cheapest first, with every rule's drop count recorded in the manifest:

  1. Source gate. Scruples: post_type == HYPOTHETICAL. WildChat: English, non-toxic, non-redacted, first turn is a user turn, and openai_moderation[0].flagged is false. WildChat redaction is a drop rather than a repair because a situation whose participants are literal "[Name]" placeholders is not a situation an assistant can advise on.
  2. Shape: character-length window (source-specific — Reddit posts run far longer than chat openers) and machine-text markers (code fences, markup, URLs, long bracketed blocks).
  3. Advice framing + first-person density: the message must ask what to do AND be about the writer. Both are required — "what should the government do about X" passes the first and fails the second.
  4. Exclusions: (a) assistant-as-tool requests (write/translate/summarize/roleplay/code), which dominate WildChat and are advice-shaped often enough to survive step 3; (b) platform-referential Reddit text; (c) the safety scope from the construction plan — abuse, self-harm, medical decisions, legal jeopardy, financial distress, and anything turning on a vulnerable minor. The safety exclusions are deliberately broad. A false positive costs one candidate out of thousands; a false negative puts a confident four-way value-divergent recommendation on a situation whose correct answer is "seek help".
  5. Instrument separation: drop anything echoing a live elicitation stimulus, via validate_sft_data.elicitation_snippets() (the same snippet list the SFT screen and the M0 builder use).
  6. Dedup: exact match on normalized text, then a per-author cap (WildChat hashed_ip) so one prolific poster cannot dominate.

Output: <out>/candidates_stage1.jsonl (one record per surviving prompt, with the rules it matched) and <out>/stage1_manifest.json (counts per rule, filter parameters, source provenance). Stage 2 (LLM dilemma classifier) and stage 3 (manual review to ~250) consume the JSONL; neither is in this script.

    python scripts/build_values_prompt_candidates.py --out data/values_prompts
    python scripts/build_values_prompt_candidates.py --source wildchat --out /tmp/wc --shards 1
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SCHEMA = "values_prompt_candidate_v1"
MANIFEST_SCHEMA = "values_prompt_candidates_stage1_manifest_v1"

SOURCES = {
    "scruples": {
        "hf_repo": "metaeval/scruples",
        "citation": "Scruples: hf.co/datasets/metaeval/scruples (arXiv:2008.09094), r/AmItheAsshole anecdotes curated by AI2",
        "license": "apache-2.0",
        # Reddit posts run long: in the HYPOTHETICAL pool the 10th/50th/90th percentiles are ~540/1400/2900 characters. A 150-1500 window (right for a chat opener) would discard most of the corpus, so the ceiling is raised. The floor is raised too: a very short AITA post is usually a one-line grievance with no situation to advise on.
        "min_chars": 300,
        "max_chars": 2200,
    },
    "wildchat": {
        "hf_repo": "allenai/WildChat-1M",
        "citation": "WildChat-1M: hf.co/datasets/allenai/WildChat-1M (arXiv:2405.01470)",
        "license": "odc-by",
        # A chat opener that carries a real situation is a paragraph or two; beyond that it is a pasted document or a multi-question dump.
        "min_chars": 150,
        "max_chars": 1500,
    },
}

# First-person density: bare pronoun hits, not distinct pronouns. A message about the writer's own situation returns to itself repeatedly; one stray "I think" in an essay request does not.
MIN_FIRST_PERSON_HITS = 3

# One author may contribute at most this many candidates. WildChat's heaviest users submit thousands of turns, and an unbounded pool would inherit one person's preoccupations as if they were the corpus's. Scruples is already deduplicated by post, so this binds only on WildChat.
MAX_PER_AUTHOR = 3

FIRST_PERSON_RE = re.compile(r"\b(i|i'm|im|i've|i'd|i'll|my|me|mine|myself|we|we're|our|us)\b", re.IGNORECASE)

# Advice framing. Each alternative has to carry the "asking what to do" force on its own — this is the only rule selecting FOR the target class, so a loose alternative here costs stage 2 real money. Deliberately no "wibta"/"aita": those are Reddit jargon, and matching on them would select exactly the posts whose ask is phrased in platform vocabulary.
ADVICE_PATTERNS = {
    "should_i": r"\bshould i\b|\bshould we\b|\bdo i (have to|need to)\b",
    "what_do_i_do": r"\bwhat (should|do|would) i do\b|\bwhat do i say\b|\bhow do i (handle|tell|approach|deal with)\b",
    "what_would_you_do": r"\bwhat would you do\b|\bwhat do you think i should\b|\bwhat would you (suggest|recommend)\b",
    "advice_request": r"\b(any|some|need|looking for|give me) advice\b|\badvise me\b|\bwhat's your advice\b",
    "undecided": r"\b(don't|do not|dont) know (whether|if|what to|how to)\b|\bnot sure (whether|if) i\b|\bcan't decide\b|\bcant decide\b|\btorn between\b|\bunsure whether\b",
    "am_i_wrong": r"\bam i (wrong|being unreasonable|overreacting|the (asshole|jerk))\b|\bwould i be (wrong|the)\b|\bwould it be (wrong|selfish|rude|bad)\b|\bis it (wrong|selfish|rude|ok|okay) (to|for me)\b",
    "help_me_decide": r"\bhelp me (decide|choose|figure out)\b|\bwhich (option|one) should i\b",
}
ADVICE_MATCHERS = {name: re.compile(pat, re.IGNORECASE) for name, pat in ADVICE_PATTERNS.items()}

# Machine text / pasted-artifact markers, checked on the raw turn so indentation and fence markers survive.
ARTIFACT_PATTERNS = {
    "code_fence": r"```",
    "markup_tag": r"</?(html|div|span|body|script|table|tr|td|p|br|h[1-6])\b",
    "url": r"https?://|www\.\w+\.",
    "code_syntax": r"\b(def |class |import |function\(|SELECT .* FROM |#include|public static|=>|\{\s*\n)",
    "bracket_block": r"\[[^\]]{40,}\]",
    "listy": r"(\n\s*[-*•]\s+.*){4,}",
}
ARTIFACT_MATCHERS = {name: re.compile(pat, re.IGNORECASE) for name, pat in ARTIFACT_PATTERNS.items()}

# Platform-referential text. Not a style screen — these make the prompt incoherent as a standalone message to an assistant, because they point at context that does not travel with it. Age-and-gender tags and "tl;dr" are deliberately NOT here: they are register, they are constant across arms, and they cost the measurement nothing.
REDDIT_META_PATTERNS = {
    "points_offsite": r"\b(post history|my last post|previous post|posted (this|about)|see my|update:|edit\s*\d*\s*:|this sub(reddit)?|r/\w+|u/\w+|\bop\b|throwaway|mobile format|long time lurker|first time post)",
    "addresses_reddit": r"\b(reddit|redditors|fellow (redditors|humans)|hive ?mind|judge me|verdict|be the asshole|yta\b|nta\b|esh\b|nah\b)",
}
REDDIT_META_MATCHERS = {name: re.compile(pat, re.IGNORECASE) for name, pat in REDDIT_META_PATTERNS.items()}

# Assistant-as-tool requests. WildChat is overwhelmingly this, and enough of it is phrased with "should I" ("should I use X in my essay") to survive the advice rule.
TASK_REQUEST_PATTERNS = {
    "write_for_me": r"\b(write|rewrite|draft|compose|generate|create|make) (me |a |an |the |this |my )?(essay|email|letter|story|poem|script|article|post|caption|code|program|paragraph|summary|outline|resume|cv|cover letter|song|chapter|novel|prompt)\b",
    "transform_text": r"\b(translate|paraphrase|summarize|summarise|proofread|correct|edit|expand|shorten|reword|continue) (this|the|my|it|following)\b|\bcheck my (grammar|spelling|code)\b",
    "roleplay": r"\b(act as|you are|pretend to be|roleplay|role-play|from now on you)\b|\bignore (all )?previous instructions\b",
    "explain_topic": r"\b(explain|what is|what are|define|list|give me) (the |a |an )?\w+ (in|of|for|to) \b.*\?|\bhow does .* work\b",
    "homework": r"\b(homework|assignment|exam|midterm|thesis|dissertation|word count|words? long|bibliography|citation)\b",
    "meta_assistant": r"\b(as an ai|language model|chatgpt|gpt-?[34]|openai|this conversation|your training)\b",
}
TASK_REQUEST_MATCHERS = {name: re.compile(pat, re.IGNORECASE) for name, pat in TASK_REQUEST_PATTERNS.items()}

# Domain screens. Which of these are ACTIVE is a design decision, set by --domains (default: romance,online) and recorded in data/values_prompts/DECISIONS.md; all four stay defined here so the choice is reversible and auditable rather than lost in a deleted regex.
#
# The construction plan originally excluded the value portraits' whole work / friendship / online triad, to keep training domains disjoint from the eval instrument. Work and friendship were dropped from the default on 2026-08-19. The contamination argument is weak: the portraits ask the model which described person it would rather BE (second person, forced choice, self-directed), while this corpus has the assistant ADVISING someone else about a first-person situation — different task, different grammatical person, different form — and literal overlap is already caught by the elicitation-stimulus screen. Against that weak risk, full disjointness costs real interpretive power: train only in family/money/school and measure on work/friendship/online portraits, and a null cannot distinguish "values did not move" from "values moved but did not transfer across domains".
#
# Romance and online stay excluded, for different reasons. Romance is a SAFETY exclusion in substance — AITA romance is dominated by infidelity and breakups, which the plan's scope excludes anyway — and is kept here only because domain vocabulary is the cheapest way to catch it. Online is excluded because it is thin on ordinary value forks and is the narrowest, most distinctive of the portrait contexts.
BANNED_DOMAIN_PATTERNS = {
    "work": r"\b(my (boss|manager|coworker|colleague|supervisor|employer|team lead)|at work|my (job|career|workplace|office|shift)|got (fired|hired)|hr department|\bclients?\b|customer service)\b",
    # Three rounds of leaks shaped this one, and each was a different failure of the same assumption — that romance announces itself with a relationship noun. It does not.
    #   (a) verbs: "this girl in my office that I like very much", debating whether to "tell her my feelings";
    #   (b) abbreviations: "my bf", "my gf", "my SO" — the spelled-out forms were listed, the short ones people actually type were not;
    #   (c) courtship phrasing with no noun and no pronoun object at all: "been flirting with", "we've even said I love you".
    "romance": r"\b(my (boyfriend|girlfriend|husband|wife|partner|spouse|fianc[\u00e9e]{1,2}|ex|bf|gf|s\.?o\.?)\b|dating|tinder|hinge|bumble|hook ?up|our relationship|broke up|breaking up|flirt\w*|(tell|told|telling) (him|her|them) (about )?my feelings|said i love you|have (a )?crush|crush on|asked (him|her) out|ask (him|her) out|romantic\w*|\bproposed?\b|feelings (for|towards) (him|her)|date (him|her)|first date|pursu\\w+ (the same |this |that )?(woman|man|girl|guy)|(woman|man|girl|guy) (i|he|she) (like|likes|fancy|fancies)|same (woman|man|girl|guy) as)",
    "friendship": r"\b(my (best )?friends?\b|friend group|friendship|my buddy|my mate)\b",
    "online": r"\b(social media|instagram|facebook|twitter|tiktok|snapchat|discord|youtube|twitch|whatsapp|snap\w*|online forum|subreddit|my post|went viral|dm(s|ed)?\b)",
}
BANNED_DOMAIN_MATCHERS = {name: re.compile(pat, re.IGNORECASE) for name, pat in BANNED_DOMAIN_PATTERNS.items()}
DEFAULT_DOMAINS = ("romance", "online")

# Safety scope from the construction plan. Deliberately over-broad: these drop candidates, and candidates are cheap.
SAFETY_PATTERNS = {
    # Paraphrase matters more than vocabulary here. The first pass matched "cutting myself" and "self-harm" and sailed past a post describing a roommate who "started scratching her arm" and "became more and more self destructive" — the exact false negative this screen exists to prevent. Third-person forms are included because the writer is usually asking about someone else.
    "self_harm": r"\b(suicid\w*|kill (myself|herself|himself|themselves)|end (my|her|his) life|self[- ]?harm\w*|self[- ]?destructive|cutting (myself|herself|himself|themselves)|scratch\w* (her|his|their|my) (arm|arms|leg|legs|thigh|thighs|skin)|burn\w* (herself|himself|themselves|myself)|overdose|want to die|hurt (myself|herself|himself|themselves)|harm\w* (herself|himself|themselves|myself))\b",
    "abuse_violence": r"\b(abus\w+|assault\w*|rape|raped|molest\w*|beat me|beats me|hit me|hits me|domestic violence|restraining order|stalk\w+|threaten\w+ me|trafficking)\b",
    # Extended after the first Scruples pass on three misses: end-of-life decisions for a pet ("putting our 18 year old blind, deaf, incontinent, newly diabetic shih-zu" down), mental-health conditions named in passing ("severe anxiety and depression"), and the general "quality of life" framing that marks a care decision rather than an ordinary choice.
    "medical": r"\b(cancer|tumou?r|chemo\w*|diagnos\w+|symptom\w*|prescri\w+|medication|antidepressant|surgery|surgeon|hospital\w*|emergency room|\ber\b|therapist|therapy|counsel(l)?or|psychiatr\w+|psycholog\w+|bipolar|schizo\w+|depress\w+|anxiety|mental (health|illness\w*|issues?|conditions?|breakdown)|relapse\w*|eating disorder|anorexi\w+|bulimi\w+|pregnan\w+|abortion|miscarriage|addict\w+|alcoholi\w+|rehab|withdrawal symptoms|chronic (pain|illness)|disabilit\w+|fever|thermometer|euthan\w+|put (him|her|them|our|my) \w* ?(down|to sleep)|quality of life|hospice|terminal\w*|dementia|alzheimer\w*|palliative)\b",
    "legal": r"\b(lawyer|attorney|lawsuit|sue (them|him|her|me)|suing|court|judge|criminal|arrest\w*|police|jail|prison|probation|deport\w+|immigration|visa|asylum|custody|divorce|restraining|illegal|felony|fraud)\b",
    "financial_distress": r"\b(evict\w+|foreclos\w+|bankrupt\w*|debt collect\w+|collections agency|garnish\w+|homeless|can't afford (rent|food|medicine)|payday loan|repossess\w+|laid off|fired|unemploy\w+)\b",
    # Widened to younger siblings and school-age framing: the first pass kept a post about a grounded younger brother with severe anxiety who had been given a vape. "school" alone is NOT here — school is an in-scope domain for this corpus — so the trigger is a named minor relative or an explicit under-18 marker.
    "minor_risk": r"\b(my (son|daughter|child|kid)s? (is|are|was|were)? ?\d{1,2}\b|\b(1[0-7]|[1-9]) years? old\b|\bminors?\b|\bunderage\b|\bcps\b|child protective|\bmy (little|younger|kid) (brother|sister|sibling)\b|\bgrounded\b|\bvap(e|ing)\b|\bmiddle school|\bhigh ?school\b|\belementary\b|\bteenager\b|\bteen\b)|[\(\[]?\b(?:1[0-7]\s?[mfMF]|[mfMF]\s?1[0-7])\b[\)\]]?",
    "sexual": r"\b(sex\w*|nsfw|porn\w*|nude|erotic|fetish|kink|virgin\w*|cheat\w+ on (me|him|her|my)|affair)\b",
    "hate_identity_conflict": r"\b(racist|racism|nazi|slur|homophob\w+|transphob\w+)\b",
}
SAFETY_MATCHERS = {name: re.compile(pat, re.IGNORECASE) for name, pat in SAFETY_PATTERNS.items()}

# Ask normalization. Scruples posts phrase their ask as the subreddit's acronym in the TITLE ("WIBTA for...") while the body only narrates the situation, so most prospective posts carry no natural-language ask anywhere in the text we keep. Matching on "wibta"/"aita" was rejected — that selects exactly the posts whose ask is platform jargon, and would drag the jargon into the corpus.
#
# The ask is recovered in order of preference:
#   1. body      — the body already asks in the poster's own words. Left exactly as written; being faithful where we can costs nothing.
#   2. title     — the title's acronym is expanded into plain English and appended. "WIBTA" is a drop-in for "Would I be wrong", and the connector the poster already used keeps it grammatical: "WIBTA for doing a separate gift?" -> "Would I be wrong for doing a separate gift?", "WIBTA if I got mad at my friend?" -> "Would I be wrong if I got mad at my friend?". This preserves the SPECIFIC question the poster asked, which a generic closer throws away.
#   3. closer    — the title does not parse into a question (no recognized connector), so a plain closer from ASK_CLOSERS is appended instead. Assigned by content hash so the result is reproducible without threading RNG state through the build.
#
# Expanding the title is not the same as prepending it. The AITA frame lives in the acronym, which is removed; what remains is a question a person could plausibly type. The title is never used as a heading or a prefix.
#
# CRITICAL: because title text can now enter the prompt, every screen runs on the COMPOSED prompt rather than the body. A title like "WIBTA if I started ghosting her?" carries romance vocabulary the body may never mention, and a screen that only ever saw the body would pass it.
#
# All of this is synthesis, not mining, and is recorded per record as ask_source so the subsets can be compared at review time. It is safe for the measurement for the same reason Reddit register is: the prompt is byte-identical across all four arms, so an appended ask is a constant and cannot differentiate them.
ASK_CLOSERS = (
    "What should I do?",
    "What would you do?",
    "I'm not sure what to do here. What do you think?",
    "Help me decide.",
)


# "WIBTA"/"AITA" plus the connector the poster used. Only these connectors are expanded: with anything else ("WIBTA pizza?", "WIBTA my brother?") the title is not a well-formed question and a generic closer is safer than a guess.
TITLE_ASK_RE = re.compile(r"^\W*(?:wibta|wibtah|aita|aitah)\b\W*(if|for|to|when|by)\b\s*(.+?)\s*\??$", re.IGNORECASE | re.DOTALL)


def ask_from_title(title: str) -> str | None:
    """Expand a subreddit-acronym title into a plain-English question, or None if it does not parse."""
    match = TITLE_ASK_RE.match((title or "").strip())
    if not match:
        return None
    connector, rest = match.group(1).lower(), match.group(2).strip()
    if not rest:
        return None
    # Lower-case a leading capital that is only there because it followed the acronym ("WIBTA If I asked" -> "if I asked"), but leave "I" and genuine proper nouns alone.
    if rest[:2] not in ("I ", "I'") and rest.split(" ", 1)[0].istitle() and not rest.startswith("I"):
        rest = rest[0].lower() + rest[1:]
    return f"Would I be wrong {connector} {rest}?"


def compose(body: str, ask: str) -> str:
    """Attach the ask as its own closing line, so it reads as the writer's last sentence rather than a splice into their final paragraph."""
    return f"{body.rstrip()}\n\n{ask}"


def fallback_closer(digest: str) -> str:
    return ASK_CLOSERS[int(digest[:8], 16) % len(ASK_CLOSERS)]


WILDCHAT_COLUMNS = ["conversation_hash", "model", "turn", "language", "toxic", "redacted", "country", "hashed_ip", "conversation", "openai_moderation"]


# Reddit bodies arrive HTML-escaped and sprinkled with zero-width joiners used as paragraph separators ("&amp;#x200B;"). Left in place these reach the training prompt as literal entity text, so they are decoded and stripped before any screen runs — a screen reading "&amp;" where the poster wrote "&" is also matching the wrong string.
# The subreddit's acronyms, wherever they appear in the body. These are NOT a drop screen: most WIBTA posts state their ask this way, so dropping them would discard the bulk of the corpus — the very posts ask-normalization exists to recover. Instead the sentence carrying the jargon is removed and a plain closer is appended in its place.
# Checked over the tail rather than the whole text: a question mark in the middle of a narrated argument is not the writer's ask.
ENDS_IN_QUESTION_RE = re.compile(r"\?[\s\)\]\"']*$")

JARGON_RE = re.compile(r"\b(wibta|aita|yta|nta|esh|iatah?)\b", re.IGNORECASE)

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]|&#x200[bB];")


def clean_source_text(text: str) -> str:
    text = html.unescape(html.unescape(text or ""))  # twice: bodies are commonly double-escaped ("&amp;#x200B;")
    text = ZERO_WIDTH_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def scrub_jargon(text: str) -> tuple[str, bool]:
    """Drop whole sentences containing subreddit acronyms. Returns (text, scrubbed).

    Sentence-level rather than word-level: excising just the token leaves "I'm tempted to refuse, but , or should I push for more information" — grammatical debris that reads worse than the jargon did. The acronym almost always sits in a dedicated closing question, so removing its sentence removes the ask and nothing else, which is precisely the state ask-normalization then repairs.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = [part for part in parts if part.strip() and not JARGON_RE.search(part)]
    if len(kept) == len(parts):
        return text, False
    return re.sub(r"\s+([,.!?])", r"\1", " ".join(kept)).strip(), True


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def hits(matchers: dict, text: str) -> list[str]:
    return [name for name, matcher in matchers.items() if matcher.search(text)]


def _snapshot(repo: str, patterns: list[str] | None = None) -> Path:
    """Resolve an already-downloaded HF snapshot. Fails loudly rather than silently building from a partial or absent download — a candidate pool quietly built from nothing would look completely normal in the manifest."""
    from huggingface_hub import snapshot_download

    kwargs = {"repo_type": "dataset", "local_files_only": True}
    if patterns:
        kwargs["allow_patterns"] = patterns
    try:
        return Path(snapshot_download(repo, **kwargs))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{repo} is not in the local HF cache ({exc}). Run: hf download {repo} --repo-type dataset")


def iter_scruples(_args) -> tuple[list[dict], Counter]:
    """Yield (text, author, meta) for HYPOTHETICAL Scruples posts across all splits. The post body only — the title is Reddit convention ('WIBTA for…') and would stamp the AITA frame onto every user turn."""
    local = _snapshot(SOURCES["scruples"]["hf_repo"])
    paths = sorted(local.glob("*.scruples-anecdotes.jsonl"))
    if not paths:
        raise SystemExit(f"no *.scruples-anecdotes.jsonl under {local} — re-download the dataset")
    drops = Counter()
    records = []
    for path in paths:
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("post_type") != "HYPOTHETICAL":
                    drops["post_type_not_hypothetical"] += 1
                    continue
                text = clean_source_text(row.get("text"))
                records.append({
                    "text": text,
                    "author": None,  # Scruples ships no author field; posts are already unique
                    # The title is carried as metadata and never enters the prompt: it is subreddit convention ("WIBTA for...") and would stamp the AITA frame onto every user turn. It is kept because it is the fastest way for a human reviewer to triage several hundred candidates, and because it is the only place the poster states their own ask in one line.
                    "meta": {"split": path.name.split(".")[0], "post_id": row.get("post_id"), "scruples_label": row.get("label"), "title": (row.get("title") or "").strip()},
                })
    return records, drops


def iter_wildchat(args) -> tuple[list[dict], Counter]:
    """Yield (text, author, meta) for the first user turn of each clean English WildChat conversation."""
    import pyarrow.parquet as pq

    local = _snapshot(SOURCES["wildchat"]["hf_repo"], ["data/*.parquet"])
    paths = sorted((local / "data").glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet shards under {local}/data")
    expected = int(paths[0].name.split("-of-")[1].split(".")[0])
    if len(paths) != expected and not args.shards:
        raise SystemExit(f"found {len(paths)} of {expected} shards — the download is incomplete")
    if args.shards:
        paths = paths[: args.shards]

    drops = Counter()
    records = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size, columns=WILDCHAT_COLUMNS):
            for row in batch.to_pylist():
                if row["language"] != "English":
                    drops["lang_not_english"] += 1
                    continue
                if row["toxic"]:
                    drops["conversation_toxic"] += 1
                    continue
                if row["redacted"]:
                    drops["redacted"] += 1
                    continue
                conversation = row["conversation"] or []
                if not conversation or conversation[0].get("role") != "user":
                    drops["no_user_first_turn"] += 1
                    continue
                first = conversation[0]
                if first.get("toxic"):
                    drops["first_turn_toxic"] += 1
                    continue
                if first.get("language") != "English":
                    drops["first_turn_not_english"] += 1
                    continue
                moderation = row["openai_moderation"] or []
                if moderation and moderation[0].get("flagged"):
                    drops["moderation_flagged"] += 1
                    continue
                records.append({
                    "text": clean_source_text(first.get("content")),
                    "author": row["hashed_ip"] or None,
                    "meta": {"shard": path.name, "conversation_hash": row["conversation_hash"], "wildchat_model": row["model"], "wildchat_turns": row["turn"], "country": row["country"]},
                })
    return records, drops


LOADERS = {"scruples": iter_scruples, "wildchat": iter_wildchat}


def build(args) -> dict:
    source = SOURCES[args.source]
    min_chars = args.min_chars or source["min_chars"]
    max_chars = args.max_chars or source["max_chars"]

    try:
        from validate_sft_data import elicitation_snippets

        snippets = elicitation_snippets()
    except Exception as exc:  # noqa: BLE001 - the screen is a nice-to-have here; its absence is recorded, not fatal
        snippets, snippet_error = [], repr(exc)
    else:
        snippet_error = None

    domains = tuple(d for d in (args.domains or "").split(",") if d.strip())
    unknown = [d for d in domains if d not in BANNED_DOMAIN_PATTERNS]
    if unknown:
        raise SystemExit(f"unknown --domains value(s) {unknown}; choose from {sorted(BANNED_DOMAIN_PATTERNS)}")
    domain_matchers = {d: BANNED_DOMAIN_MATCHERS[d] for d in domains}

    records, drops = LOADERS[args.source](args)
    rows_read = len(records) + sum(drops.values())

    kept_rule_counts = Counter()
    per_author = Counter()
    ask_sources = Counter()
    seen = set()
    candidates = []

    for record in records:
        body = record["text"]

        if not (min_chars <= len(body) <= max_chars):
            drops["length_window"] += 1
            continue

        body, scrubbed = scrub_jargon(body)
        if scrubbed and not (min_chars <= len(body) <= max_chars):
            drops["length_window_after_jargon_scrub"] += 1
            continue

        # Resolve the ask before screening, because the ask may bring in text (a title) the body never contained.
        digest = hashlib.sha256(normalize(body).encode()).hexdigest()
        advice = hits(ADVICE_MATCHERS, body)
        if not advice and ENDS_IN_QUESTION_RE.search(body):
            advice = ["trailing_question"]
        title_ask = ask_from_title(record["meta"].get("title", "")) if args.normalize_ask else None
        if title_ask and (args.title_ask_everywhere or not advice):
            # The de-jargoned title question is the last phrase of the prompt. With
            # --title_ask_everywhere (the default) that holds for every record whose title
            # parses, including ones whose body already asks: the body's own question is
            # usually vague ("what should I do?") while the title states the specific fork,
            # and a uniform closing question is worth the occasional two-questions-in-a-row.
            ask_source = "body+title" if advice else "title"
            prompt = compose(body, title_ask)
        elif advice:
            ask_source, prompt = "body", body
        elif args.normalize_ask:
            ask_source, prompt = "closer", compose(body, fallback_closer(digest))
        else:
            drops["no_advice_framing"] += 1
            continue

        # Every screen from here runs on the COMPOSED prompt, not the body.
        artifact = hits(ARTIFACT_MATCHERS, prompt)
        if artifact:
            drops[f"artifact:{artifact[0]}"] += 1
            continue
        if len(FIRST_PERSON_RE.findall(prompt)) < MIN_FIRST_PERSON_HITS:
            drops["not_first_person"] += 1
            continue
        task = hits(TASK_REQUEST_MATCHERS, prompt)
        if task:
            drops[f"task_request:{task[0]}"] += 1
            continue
        meta_ref = hits(REDDIT_META_MATCHERS, prompt)
        if meta_ref:
            drops[f"platform_reference:{meta_ref[0]}"] += 1
            continue
        if JARGON_RE.search(prompt):
            drops["jargon_survived_composition"] += 1
            continue
        banned_domain = hits(domain_matchers, prompt)
        if banned_domain:
            drops[f"banned_domain:{banned_domain[0]}"] += 1
            continue
        safety = hits(SAFETY_MATCHERS, prompt)
        if safety:
            drops[f"safety:{safety[0]}"] += 1
            continue

        norm = normalize(prompt)
        if any(s in norm for s in snippets):
            drops["elicitation_stimulus_echo"] += 1
            continue

        prompt_digest = hashlib.sha256(norm.encode()).hexdigest()
        if prompt_digest in seen:
            drops["duplicate_prompt"] += 1
            continue
        author = record["author"]
        if author and per_author[author] >= MAX_PER_AUTHOR:
            drops["author_cap"] += 1
            continue

        seen.add(prompt_digest)
        if author:
            per_author[author] += 1
        for rule in advice:
            kept_rule_counts[rule] += 1
        ask_sources[ask_source] += 1
        if scrubbed:
            ask_sources["jargon_scrubbed"] += 1
        candidates.append({
            "schema": SCHEMA,
            "candidate_id": prompt_digest[:16],
            "source": args.source,
            "prompt": prompt,
            "chars": len(prompt),
            "ask_source": ask_source,
            "jargon_scrubbed": scrubbed,
            "advice_rules": advice,
            "first_person_hits": len(FIRST_PERSON_RE.findall(prompt)),
            "meta": {"citation": source["citation"], "license": source["license"], **record["meta"]},
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "candidates_stage1.jsonl"
    with jsonl_path.open("w") as fh:
        for record in candidates:
            fh.write(json.dumps(record) + "\n")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": args.source,
        "source_hf": f"https://huggingface.co/datasets/{source['hf_repo']}",
        "source_citation": source["citation"],
        "license": source["license"],
        "selection": ("post_type == HYPOTHETICAL (Scruples' own prospective/retrospective label); post body only, title excluded"
                      if args.source == "scruples" else "first user turn only (conversation[0].content)"),
        "parameters": {
            "domains_excluded": list(domains),
            "normalize_ask": args.normalize_ask,
            "ask_closers": list(ASK_CLOSERS),
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_first_person_hits": MIN_FIRST_PERSON_HITS,
            "max_per_author": MAX_PER_AUTHOR,
        },
        "patterns": {
            "advice": ADVICE_PATTERNS,
            "artifact": ARTIFACT_PATTERNS,
            "task_request": TASK_REQUEST_PATTERNS,
            "platform_reference": REDDIT_META_PATTERNS,
            "banned_domain": BANNED_DOMAIN_PATTERNS,
            "safety": SAFETY_PATTERNS,
        },
        "elicitation_snippets_loaded": len(snippets),
        "elicitation_snippet_error": snippet_error,
        "counts": {
            "rows_read": rows_read,
            "reached_screens": len(records),
            "candidates": len(candidates),
            "ask_source": dict(ask_sources),
            "yield_pct_of_screened": round(100 * len(candidates) / len(records), 2) if records else None,
        },
        "drops": dict(sorted(drops.items(), key=lambda kv: -kv[1])),
        "kept_by_advice_rule": dict(sorted(kept_rule_counts.items(), key=lambda kv: -kv[1])),
    }
    (out_dir / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCES), default="scruples")
    ap.add_argument("--out", default=str(ROOT / "data" / "values_prompts"), help="output directory for candidates_stage1.jsonl + stage1_manifest.json")
    ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS), help=f"comma-separated domain screens to APPLY; choose from {sorted(BANNED_DOMAIN_PATTERNS)}, or '' for none. Default {','.join(DEFAULT_DOMAINS)} — see data/values_prompts/DECISIONS.md")
    ap.add_argument("--title_ask_everywhere", action=argparse.BooleanOptionalAction, default=True, help="end every prompt whose title parses with the de-jargoned title question, even when the body already asks; --no-title_ask_everywhere restores body-ask precedence")
    ap.add_argument("--normalize_ask", action=argparse.BooleanOptionalAction, default=True, help="keep posts whose body never asks anything by appending a closer (see ASK_CLOSERS); --no-normalize_ask drops them instead")
    ap.add_argument("--min_chars", type=int, default=None, help="override the source's length floor")
    ap.add_argument("--max_chars", type=int, default=None, help="override the source's length ceiling")
    ap.add_argument("--shards", type=int, default=None, help="wildchat only: use just the first N parquet shards (smoke runs)")
    ap.add_argument("--batch_size", type=int, default=2000, help="wildchat only: parquet row-batch size")
    args = ap.parse_args()

    manifest = build(args)
    counts = manifest["counts"]
    print(f"source={manifest['source']}  rows_read={counts['rows_read']:,}  reached_screens={counts['reached_screens']:,}  candidates={counts['candidates']:,}  ({counts['yield_pct_of_screened']}% of screened)")
    print(f"ask_source: {counts['ask_source']}")
    print("top drops:")
    for rule, n in list(manifest["drops"].items())[:14]:
        print(f"  {rule:34s} {n:>9,}")


if __name__ == "__main__":
    main()
