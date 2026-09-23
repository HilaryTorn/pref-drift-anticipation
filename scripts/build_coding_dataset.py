"""Build a language-balanced subset of the Magicoder-OSS-Instruct-75K dataset."""

import argparse
from pathlib import Path

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset


def build_balanced_coding_dataset(cfg: dict) -> Dataset:
    """Download the dataset and return a language-balanced subset.

    Args:
        cfg: The coding dataset config (datasets.coding in config.yaml).

    Returns:
        A HuggingFace Dataset with an equal number of examples per language.
    """
    languages = cfg["languages"]
    cap = cfg["cap"]

    ds = load_dataset(cfg["repo_id"], split=cfg["split"])

    subsets = []
    for lang in languages:
        subset = ds.filter(lambda x, lang=lang: x["lang"] == lang)

        available = len(subset)
        if available < cap:
            raise ValueError(
                f"Language '{lang}' has only {available} examples, "
                f"fewer than the requested cap of {cap}."
            )

        if cfg.get("shuffle", True):
            subset = subset.shuffle(seed=cfg.get("seed", 42))

        subset = subset.select(range(cap))
        subsets.append(subset)
        print(f"{lang}: kept {cap} of {available}")

    balanced = concatenate_datasets(subsets)

    keep_columns = cfg.get("keep_columns")
    if keep_columns:
        balanced = balanced.select_columns(keep_columns)

    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a language-balanced coding dataset."
    )
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--config",
        default=str(root / "config.yaml"),
        help="Path to the YAML config file (default: <root>/config.yaml).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg = config["datasets"]["coding"]

    balanced = build_balanced_coding_dataset(cfg)
    print(f"Total examples: {len(balanced)}")
    print(f"Columns: {balanced.column_names}")

    out_path = root / cfg["path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_json(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
