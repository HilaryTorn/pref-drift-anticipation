#!/usr/bin/env python3
"""Utilities for ternary anticipation alpha calibration and drift labeling.

This module intentionally stays small: live scoring produces forecasts, utility
scoring produces pre/post means, and this helper defines the pre-registered
no-change band used to compare them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
DriftLabel = Literal["MORE", "LESS", "SAME"]


def load_tau_rule(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text()) or {}
    try:
        rule = config["analysis"]["anticipation_alpha"]
    except KeyError as exc:
        raise ValueError(f"{config_path} is missing analysis.anticipation_alpha") from exc

    tau = rule.get("tau")
    if not isinstance(tau, dict):
        raise ValueError(f"{config_path} analysis.anticipation_alpha.tau must be a mapping")
    required = {
        "status",
        "unit",
        "calibration_method",
        "min_replicates_per_book",
        "estimator",
        "formula",
        "required_before_reporting_alpha",
    }
    missing = sorted(required - set(tau))
    if missing:
        raise ValueError(f"{config_path} tau rule missing fields: {missing}")
    if tau["calibration_method"] != "repeated_no_training_reruns":
        raise ValueError("tau calibration_method must remain repeated_no_training_reruns")
    if tau["estimator"] != "q95_abs_delta":
        raise ValueError("tau estimator must remain q95_abs_delta unless the method is re-registered")
    if int(tau["min_replicates_per_book"]) < 2:
        raise ValueError("tau calibration needs at least two no-training replicates")
    if tau["required_before_reporting_alpha"] is not True:
        raise ValueError("alpha reporting must require a calibrated tau")
    return rule


def label_observed_drift(delta: float, tau: float) -> DriftLabel:
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if delta > tau:
        return "MORE"
    if delta < -tau:
        return "LESS"
    return "SAME"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    rule = load_tau_rule(args.config)
    if args.delta is not None or args.tau is not None:
        if args.delta is None or args.tau is None:
            raise SystemExit("--delta and --tau must be provided together")
        print(label_observed_drift(args.delta, args.tau))
    else:
        print(json.dumps(rule, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
