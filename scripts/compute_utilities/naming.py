"""Output-file naming helpers shared by the scoring / elicitation scripts.

Dependency-free (stdlib only) on purpose: importing this must never pull in the
heavy inference stack, since the scripts import it only to name their result files.
"""
from datetime import datetime, timezone


def run_timestamp() -> str:
    """Compact **UTC** timestamp like ``20260709T142530Z``.

    Always UTC, never local time: results are pooled across teammates in different
    time zones, so a local-time stamp would sort wrong and could collide ambiguously.
    The trailing ``Z`` makes the UTC explicit in the filename.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def timestamped(suffix: str, enabled: bool = True) -> str:
    """Append a run timestamp to an output-file ``suffix`` unless disabled.

    ``enabled`` is wired to ``not args.no_timestamp`` in each script, so the default
    (timestamp on) means re-running never silently overwrites a prior run's files.
    """
    return f"{suffix}_{run_timestamp()}" if enabled else suffix


def default_results_dir(model_key: str, category: str, base: str = "results") -> str:
    """Default output directory: ``results/<model_key>/<category>``.

    Keeps every run for one model (a base or a checkpoint) together, and each battery
    type (``pairs`` / ``ue`` / ``anticipation`` / ``label_variants``) in its own
    subfolder — so a full trajectory browses cleanly instead of piling flat into
    ``results/``. Scripts fall back to this only when ``--save_dir`` isn't given.
    """
    return f"{base}/{model_key}/{category}"
