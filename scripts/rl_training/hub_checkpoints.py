"""Live Hugging Face archival for formal training checkpoints.

The callback uploads model/adapter checkpoints immediately after every save.
Optimizer, scheduler, scaler, and RNG state are intentionally excluded. A
checkpoint is eligible for local pruning only after the Hub commit succeeds
and every uploaded path is visible at that immutable commit. Failed uploads
stay on disk. Transient failures are retried at the next save; persistent Hub
quota failures switch the run to local fallback so they do not stall every
subsequent checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from transformers import TrainerCallback


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
LOCAL_FALLBACK_MARKER = ".hub_uploads_disabled"
MODEL_ONLY_IGNORE_PATTERNS = (
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
)
PERSISTENT_HUB_FAILURE_MARKERS = (
    "private repository storage limit reached",
    "storage quota exceeded",
    "storage limit reached",
)


def is_persistent_hub_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in PERSISTENT_HUB_FAILURE_MARKERS)


def is_training_state_file(path: Path) -> bool:
    return any(path.match(pattern) for pattern in MODEL_ONLY_IGNORE_PATTERNS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_manifest_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"frozen cohort manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid frozen cohort manifest JSON: {path}: {exc}") from exc
    if (
        manifest.get("schema") != "shared_training_cohort_v1"
        or manifest.get("status") != "frozen"
        or manifest.get("selection", {}).get("selected_size") != 1000
    ):
        raise ValueError(
            "formal Hub upload requires a frozen shared_training_cohort_v1 "
            "with exactly 1,000 selected prompts"
        )
    return sha256_file(path)


def formal_hub_path_prefix(
    *, base_model_id: str, base_revision: str, cohort_sha256: str, algo: str, seed: int
) -> str:
    model_slug = re.sub(r"[^a-z0-9]+", "-", base_model_id.lower()).strip("-")
    return (
        f"base-{model_slug}--rev-{base_revision[:12]}/"
        f"cohort-{cohort_sha256[:12]}/{algo}/seed-{seed}"
    )


def validate_formal_hub_settings(
    *,
    output_dir: Path,
    repo_id: str,
    base_model_id: str,
    base_revision: str,
    cohort_manifest: Path,
    algo: str,
    seed: int | None = None,
) -> str:
    resolved_output = output_dir.resolve()
    lowered = str(resolved_output).lower()
    if lowered == "/tmp" or lowered.startswith("/tmp/") or "smoke" in lowered:
        raise ValueError(f"refusing to upload a smoke or temporary run: {resolved_output}")
    if "smoke" in repo_id.lower():
        raise ValueError(f"refusing to upload to a smoke repository: {repo_id}")
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.endswith("/"):
        raise ValueError(f"--hub_repo_id must be an org/name model repository: {repo_id}")
    if "/" not in base_model_id or Path(base_model_id).is_absolute():
        raise ValueError("--hub_base_model_id must be the canonical Hugging Face model id")
    if not IMMUTABLE_REVISION_RE.fullmatch(base_revision):
        raise ValueError("--hub_base_revision must be an immutable 40-character commit SHA")
    if algo not in {"dpo", "grpo", "ppo", "sft"}:
        raise ValueError(f"unsupported formal algorithm: {algo}")
    cohort_path = cohort_manifest.resolve()
    cohort_hash = cohort_manifest_sha256(cohort_path)
    if seed is not None:
        manifest = json.loads(cohort_path.read_text())
        if manifest.get("selection", {}).get("seed") != seed:
            raise ValueError("formal Hub upload seed differs from the frozen cohort seed")
    return cohort_hash


def resolve_hub_token(token_env: str) -> str:
    from huggingface_hub import get_token

    token = os.environ.get(token_env) or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise ValueError(
            f"no Hugging Face token found; set {token_env} or HF_TOKEN, or run `hf auth login`"
        )
    return token


def preflight_hub_repo(*, repo_id: str, private: bool, token: str) -> Any:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    identity = api.whoami()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    print(
        f"[hub] authentication OK as {identity.get('name', '<unknown>')}; "
        f"formal checkpoints -> {repo_id}"
    )
    return api


class FormalHubCheckpointCallback(TrainerCallback):
    """Upload complete formal checkpoints and bound local checkpoint storage."""

    def __init__(
        self,
        *,
        repo_id: str,
        path_prefix: str,
        algo: str,
        seed: int,
        base_model_id: str,
        base_revision: str,
        cohort_sha256: str,
        tokenizer: Any,
        token: str | None = None,
        private: bool = True,
        keep_local_checkpoints: int = 1,
        api: Any | None = None,
    ) -> None:
        if keep_local_checkpoints < 0:
            raise ValueError("keep_local_checkpoints must be non-negative")
        self.repo_id = repo_id
        self.path_prefix = path_prefix.strip("/")
        self.algo = algo
        self.seed = seed
        self.base_model_id = base_model_id
        self.base_revision = base_revision
        self.cohort_sha256 = cohort_sha256
        self.tokenizer = tokenizer
        self.token = token
        self.private = private
        self.keep_local_checkpoints = keep_local_checkpoints
        self.api = api
        self._uploaded_steps: set[int] = set()
        self._loaded_upload_logs: set[Path] = set()
        self._reported_local_fallback = False

    @staticmethod
    def _world_zero(state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    @staticmethod
    def _checkpoint_dirs(output_dir: Path) -> list[tuple[int, Path]]:
        found: list[tuple[int, Path]] = []
        for candidate in output_dir.glob("checkpoint-*"):
            match = CHECKPOINT_RE.fullmatch(candidate.name)
            if match and candidate.is_dir():
                found.append((int(match.group(1)), candidate))
        return sorted(found)

    def _api(self) -> Any:
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi(token=self.token)
        return self.api

    def _normalize_adapter_provenance(self, checkpoint_dir: Path) -> None:
        """Replace machine-local PEFT provenance with the pinned Hub model.

        PEFT derives both ``adapter_config.json`` and the model-card YAML from
        ``model.name_or_path``. Formal jobs load an immutable local snapshot,
        so the generated values are absolute paths that the Hub rejects as
        invalid ``base_model`` metadata and that would not be portable.
        """
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if adapter_config_path.is_file():
            adapter_config = json.loads(adapter_config_path.read_text())
            adapter_config["base_model_name_or_path"] = self.base_model_id
            adapter_config["revision"] = self.base_revision
            adapter_config_path.write_text(
                json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
            )

        readme_path = checkpoint_dir / "README.md"
        if not readme_path.is_file():
            return
        readme = readme_path.read_text()
        match = re.match(r"\A---\s*\n(?P<metadata>.*?)\n---\s*\n", readme, re.DOTALL)
        if match is None:
            return
        metadata_lines: list[str] = []
        for line in match.group("metadata").splitlines():
            if re.fullmatch(r"base_model:\s*.*", line):
                line = f"base_model: {self.base_model_id}"
            elif re.fullmatch(r"\s*-\s*base_model:adapter:.*", line):
                indentation = line[: len(line) - len(line.lstrip())]
                line = f"{indentation}- base_model:adapter:{self.base_model_id}"
            metadata_lines.append(line)
        normalized = "---\n" + "\n".join(metadata_lines) + "\n---\n" + readme[match.end() :]
        readme_path.write_text(normalized)

    def _write_manifest(self, checkpoint_dir: Path, step: int, artifact_type: str) -> list[str]:
        self._normalize_adapter_provenance(checkpoint_dir)
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(checkpoint_dir.rglob("*")):
            relative_path = path.relative_to(checkpoint_dir)
            if (
                path.is_file()
                and path.name != "hub_checkpoint_manifest.json"
                and not is_training_state_file(relative_path)
            ):
                relative = relative_path.as_posix()
                files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema": "formal_hub_checkpoint_v1",
            "artifact_type": artifact_type,
            "algorithm": self.algo,
            "seed": self.seed,
            "optimizer_step": step,
            "base_model": self.base_model_id,
            "base_model_revision": self.base_revision,
            "cohort_manifest_sha256": self.cohort_sha256,
            "optimizer_state_uploaded": "optimizer.pt" in files,
            "files": files,
        }
        manifest_path = checkpoint_dir / "hub_checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return sorted([*files, manifest_path.name])

    def _remote_checkpoint_path(self, step: int) -> str:
        return f"{self.path_prefix}/checkpoint-{step}"

    def _record_upload(
        self, output_dir: Path, *, step: int, commit_oid: str, remote_path: str, artifact_type: str
    ) -> None:
        record = {
            "schema": "formal_hub_upload_event_v1",
            "step": step,
            "artifact_type": artifact_type,
            "repo_id": self.repo_id,
            "path_in_repo": remote_path,
            "commit_oid": commit_oid,
        }
        with (output_dir / "hub_upload_log.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _upload_directory(
        self, output_dir: Path, checkpoint_dir: Path, step: int, artifact_type: str
    ) -> bool:
        remote_path = self._remote_checkpoint_path(step)
        api = self._api()
        try:
            relative_files = self._write_manifest(checkpoint_dir, step, artifact_type)
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
            )
            result = api.upload_folder(
                folder_path=str(checkpoint_dir),
                ignore_patterns=list(MODEL_ONLY_IGNORE_PATTERNS),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=(
                    f"Archive {self.algo.upper()} seed-{self.seed} checkpoint step {step}"
                ),
            )
            commit_oid = str(getattr(result, "oid", "") or "")
            if not commit_oid:
                raise RuntimeError("Hub upload returned no immutable commit oid")
            remote_files = set(
                api.list_repo_files(
                    self.repo_id,
                    revision=commit_oid,
                    repo_type="model",
                )
            )
            expected = {f"{remote_path}/{relative}" for relative in relative_files}
            missing = sorted(expected - remote_files)
            if missing:
                raise RuntimeError(
                    f"Hub commit {commit_oid} is missing {len(missing)} uploaded files; "
                    f"first missing path: {missing[0]}"
                )
        except Exception as exc:
            persistent_failure = is_persistent_hub_failure(exc)
            if persistent_failure:
                marker_path = output_dir / LOCAL_FALLBACK_MARKER
                marker_path.write_text(
                    "Automatic local fallback after persistent Hugging Face upload "
                    f"failure: {exc}\n"
                )
            print(
                f"WARNING: Hub upload/verification failed for {checkpoint_dir}; "
                "keeping it locally"
                + (
                    f" and disabling retries via {LOCAL_FALLBACK_MARKER}: {exc}"
                    if persistent_failure
                    else f" and retrying at the next save: {exc}"
                ),
                file=sys.stderr,
            )
            return False
        self._uploaded_steps.add(step)
        self._record_upload(
            output_dir,
            step=step,
            commit_oid=commit_oid,
            remote_path=remote_path,
            artifact_type=artifact_type,
        )
        print(f"[hub] verified step {step}: hf://{self.repo_id}/{remote_path}@{commit_oid}")
        return True

    def _upload_pending(self, output_dir: Path) -> None:
        if (output_dir / LOCAL_FALLBACK_MARKER).is_file():
            if not self._reported_local_fallback:
                print(
                    f"[hub] {LOCAL_FALLBACK_MARKER} present; preserving all new checkpoints "
                    "locally without further upload attempts"
                )
                self._reported_local_fallback = True
            return
        resolved_output = output_dir.resolve()
        if resolved_output not in self._loaded_upload_logs:
            log_path = output_dir / "hub_upload_log.jsonl"
            if log_path.is_file():
                for line in log_path.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        event.get("schema") == "formal_hub_upload_event_v1"
                        and event.get("repo_id") == self.repo_id
                        and str(event.get("path_in_repo", "")).startswith(
                            f"{self.path_prefix}/checkpoint-"
                        )
                        and isinstance(event.get("step"), int)
                    ):
                        self._uploaded_steps.add(event["step"])
            self._loaded_upload_logs.add(resolved_output)
        for step, checkpoint_dir in self._checkpoint_dirs(output_dir):
            if step in self._uploaded_steps:
                continue
            if not self._upload_directory(
                output_dir, checkpoint_dir, step, "model_only_training_checkpoint"
            ):
                break

    def _prune(self, output_dir: Path, keep: int) -> None:
        uploaded = [
            (step, path)
            for step, path in self._checkpoint_dirs(output_dir)
            if step in self._uploaded_steps
        ]
        prune = uploaded[:-keep] if keep else uploaded
        resolved_root = output_dir.resolve()
        for step, path in prune:
            resolved = path.resolve()
            if resolved.parent != resolved_root or not CHECKPOINT_RE.fullmatch(resolved.name):
                raise RuntimeError(f"refusing unsafe checkpoint prune: {resolved}")
            shutil.rmtree(resolved)
            print(f"[hub] removed verified local checkpoint-{step}")

    def archive_final(self, output_dir: Path, step: int) -> bool:
        """Archive the final root adapter after Trainer.save_model has completed."""
        if (output_dir / LOCAL_FALLBACK_MARKER).is_file():
            self._prune(output_dir, self.keep_local_checkpoints)
            return False
        if step in self._uploaded_steps:
            self._prune(output_dir, 0)
            return True
        with tempfile.TemporaryDirectory(prefix=f"{self.algo}-final-adapter-") as tmp:
            staging = Path(tmp)
            for path in output_dir.iterdir():
                if path.is_file() and path.name not in {
                    "hub_upload_log.jsonl",
                    "run_metadata.json",
                    "training_manifest.json",
                }:
                    shutil.copy2(path, staging / path.name)
            if not (staging / "adapter_config.json").is_file():
                print(
                    f"WARNING: final adapter is missing at {output_dir}; keeping the newest "
                    "model checkpoint locally",
                    file=sys.stderr,
                )
                return False
            uploaded = self._upload_directory(
                output_dir, staging, step, "final_inference_adapter"
            )
        self._prune(output_dir, 0 if uploaded else self.keep_local_checkpoints)
        return uploaded

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self._world_zero(state):
            return control
        output_dir = Path(args.output_dir)
        self._upload_pending(output_dir)
        self._prune(output_dir, self.keep_local_checkpoints)
        return control

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._world_zero(state):
            return control
        output_dir = Path(args.output_dir)
        self._upload_pending(output_dir)
        # Trainer.save_model runs after this callback in our entry points. Keep
        # the newest model checkpoint until archive_final() confirms step 125.
        self._prune(output_dir, self.keep_local_checkpoints)
        return control


def add_formal_hub_arguments(parser: Any, *, include_cohort_manifest: bool = True) -> None:
    group = parser.add_argument_group("formal Hugging Face checkpoint archive")
    group.add_argument(
        "--hub_repo_id",
        default=None,
        help="Model repo receiving live formal checkpoints; omit to disable uploads",
    )
    group.add_argument(
        "--hub_base_model_id",
        default=None,
        help="Canonical base model id recorded in the remote checkpoint provenance",
    )
    group.add_argument(
        "--hub_base_revision",
        default=None,
        help="Immutable 40-character base model commit used by this run",
    )
    if include_cohort_manifest:
        group.add_argument(
            "--cohort_manifest",
            type=Path,
            default=None,
            help="Frozen exact-1,000 shared cohort manifest used for Hub namespacing",
        )
    group.add_argument(
        "--hub_token_env",
        default="HUGGINGFACE_HUB_TOKEN",
        help="Environment variable containing the write token (fallback: HF_TOKEN/login)",
    )
    group.add_argument(
        "--hub_public",
        action="store_false",
        dest="hub_private",
        help="Create the checkpoint repository as public (default: private)",
    )
    group.set_defaults(hub_private=True)
    group.add_argument(
        "--hub_keep_local_checkpoints",
        type=int,
        default=1,
        help="Number of newest verified checkpoints kept during training (default: 1; 0 after completion)",
    )


def prepare_formal_hub_upload(args: Any, *, output_dir: Path, algo: str) -> None:
    """Validate/authenticate before model loading and cache prepared state on args."""
    args._formal_hub = None
    if not args.hub_repo_id:
        supplied = [args.hub_base_model_id, args.hub_base_revision]
        if any(value is not None for value in supplied):
            raise ValueError("--hub_repo_id is required when any formal Hub option is supplied")
        return
    missing = [
        name
        for name in ("hub_base_model_id", "hub_base_revision", "cohort_manifest")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError("formal Hub upload is missing: " + ", ".join(f"--{name}" for name in missing))
    if args.hub_keep_local_checkpoints < 0:
        raise ValueError("--hub_keep_local_checkpoints must be non-negative")
    cohort_hash = validate_formal_hub_settings(
        output_dir=output_dir,
        repo_id=args.hub_repo_id,
        base_model_id=args.hub_base_model_id,
        base_revision=args.hub_base_revision,
        cohort_manifest=args.cohort_manifest,
        algo=algo,
        seed=args.seed,
    )
    if algo != "sft":
        manifest = json.loads(args.cohort_manifest.read_text())
        if algo == "dpo":
            expected = (
                manifest.get("outputs", {}).get("artifacts", {}).get("dpo", {}).get("sha256")
            )
        else:
            expected = manifest.get("outputs", {}).get("train", {}).get("sha256")
        actual = sha256_file(Path(args.dataset))
        if not expected or actual != expected:
            raise ValueError(
                f"{algo.upper()} training dataset does not match the hash-pinned frozen cohort"
            )
    token = resolve_hub_token(args.hub_token_env)
    try:
        api = preflight_hub_repo(
            repo_id=args.hub_repo_id,
            private=args.hub_private,
            token=token,
        )
    except Exception as exc:
        raise ValueError(
            f"authentication or write preflight failed for {args.hub_repo_id}: {exc}"
        ) from exc
    path_prefix = formal_hub_path_prefix(
        base_model_id=args.hub_base_model_id,
        base_revision=args.hub_base_revision,
        cohort_sha256=cohort_hash,
        algo=algo,
        seed=args.seed,
    )
    args._formal_hub = {
        "cohort_sha256": cohort_hash,
        "path_prefix": path_prefix,
        "token": token,
        "api": api,
    }
    print(f"[hub] path namespace: {path_prefix}/checkpoint-<step>")


def prepared_formal_hub_callback(args: Any, tokenizer: Any) -> FormalHubCheckpointCallback | None:
    prepared = getattr(args, "_formal_hub", None)
    if prepared is None:
        return None
    return FormalHubCheckpointCallback(
        repo_id=args.hub_repo_id,
        path_prefix=prepared["path_prefix"],
        algo=args.algo,
        seed=args.seed,
        base_model_id=args.hub_base_model_id,
        base_revision=args.hub_base_revision,
        cohort_sha256=prepared["cohort_sha256"],
        tokenizer=tokenizer,
        token=prepared["token"],
        private=args.hub_private,
        keep_local_checkpoints=args.hub_keep_local_checkpoints,
        api=prepared["api"],
    )
