"""Dataset adapter registry.

Each adapter converts one raw dataset into canonical ``coding_task_v1`` rows.
Keep dataset-specific column handling out of training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scripts.rl_training.data.schema import CodingTaskRecord


class CodingDatasetAdapter(Protocol):
    dataset_id: str
    dataset_name: str
    default_split: str

    def extract_record(self, row: dict, *, max_tests: int) -> CodingTaskRecord | None:
        """Return a canonical record, or None if the raw row is unusable."""


@dataclass(frozen=True)
class AdapterInfo:
    dataset_id: str
    dataset_name: str
    default_split: str


_ADAPTERS: dict[str, CodingDatasetAdapter] = {}


def register_adapter(adapter: CodingDatasetAdapter) -> CodingDatasetAdapter:
    if adapter.dataset_id in _ADAPTERS:
        raise ValueError(f"duplicate dataset adapter {adapter.dataset_id!r}")
    _ADAPTERS[adapter.dataset_id] = adapter
    return adapter


def get_adapter(dataset_id: str) -> CodingDatasetAdapter:
    _ensure_builtin_adapters()
    try:
        return _ADAPTERS[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS)) or "<none>"
        raise KeyError(f"unknown dataset_id {dataset_id!r}; available: {available}") from exc


def list_adapters() -> list[AdapterInfo]:
    _ensure_builtin_adapters()
    return [
        AdapterInfo(
            dataset_id=adapter.dataset_id,
            dataset_name=adapter.dataset_name,
            default_split=adapter.default_split,
        )
        for adapter in sorted(_ADAPTERS.values(), key=lambda item: item.dataset_id)
    ]


def _ensure_builtin_adapters() -> None:
    # Import for registration side effects. Keep this lazy so importing schema
    # utilities never imports optional dataset-loading dependencies.
    from scripts.rl_training.data import nemotron  # noqa: F401
