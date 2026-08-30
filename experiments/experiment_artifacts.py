"""Shared artifact and provenance contract for exploratory experiments.

The formal Fisher and Allen--Cahn runners predate this module and remain
unchanged because their source hashes are frozen into existing checkpoints.
New exploratory evaluators use this module instead of each reimplementing
path guards, atomic serialization, SHA-256 checks, and environment receipts.

The interface is intentionally small:

* reserve one brand-new exploratory root;
* hash and re-hash immutable inputs;
* write strict JSON, CSV, and Torch artifacts atomically;
* record source and runtime provenance.

No function in this module deletes, overwrites, trains, or selects a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


FORBIDDEN_OUTPUT_COMPONENTS = frozenset(
    {
        "formal",
        "locked",
        "results",
        "results-recovery",
        "results_recovery",
        "checkpoints",
        "source.tar.gz",
    }
)


def resolve_path(path: os.PathLike[str] | str) -> Path:
    """Return an expanded absolute path without requiring it to exist."""

    return Path(path).expanduser().resolve()


def require_input_file(path: os.PathLike[str] | str, label: str) -> Path:
    """Resolve one read-only input and fail before evaluation if it is absent."""

    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def assert_new_exploratory_root(
    output_dir: os.PathLike[str] | str,
    *,
    input_paths: Sequence[os.PathLike[str] | str] = (),
) -> Path:
    """Validate a unique output root that cannot overlap protected inputs.

    The check deliberately rejects every existing path, including an empty
    directory.  A failed run therefore remains an immutable, inspectable root
    and a retry must use a different canonical name.
    """

    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError(
            "output directory must be a new ordinary directory, not a symlink: "
            f"{raw_output}"
        )
    resolved = resolve_path(raw_output)
    lowered_parts = {part.lower() for part in resolved.parts}
    forbidden = sorted(lowered_parts.intersection(FORBIDDEN_OUTPUT_COMPONENTS))
    if forbidden:
        raise ValueError(
            "output directory must be a new exploratory root, not a "
            "formal/locked/results/checkpoint path: "
            f"{resolved} (forbidden components={forbidden})"
        )
    for raw_input in input_paths:
        input_path = resolve_path(raw_input)
        if resolved == input_path:
            raise ValueError("output directory may not equal an input artifact")
        if resolved in input_path.parents or input_path in resolved.parents:
            raise ValueError(
                "output directory must not be an ancestor/descendant of an input "
                f"artifact: output={resolved}, input={input_path}"
            )
    if resolved.exists():
        raise ValueError(
            "output directory must be a new exploratory root; refusing to reuse "
            f"existing path: {resolved}"
        )
    return resolved


def reserve_exploratory_root(
    output_dir: os.PathLike[str] | str,
    *,
    input_paths: Sequence[os.PathLike[str] | str] = (),
) -> Path:
    """Validate and atomically reserve a brand-new exploratory root."""

    root = assert_new_exploratory_root(output_dir, input_paths=input_paths)
    root.mkdir(parents=True, exist_ok=False)
    return root


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Compute the SHA-256 digest of one file using bounded memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_unchanged(
    path: os.PathLike[str] | str,
    before: str,
    label: str,
) -> str:
    """Re-hash one input and fail if checkpoint-only evaluation changed it."""

    after = sha256_file(path)
    if after != before:
        raise RuntimeError(
            f"{label} changed during checkpoint-only evaluation: "
            f"before={before}, after={after}"
        )
    return after


def git_commit(repository_root: os.PathLike[str] | str) -> str | None:
    """Return the current Git commit without making a repository mutation."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolve_path(repository_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def source_hashes(paths: Mapping[str, os.PathLike[str] | str]) -> dict[str, str]:
    """Hash every existing source in a named reconstruction contract."""

    return {
        str(name): sha256_file(path)
        for name, path in paths.items()
        if Path(path).is_file()
    }


def device_provenance(device: torch.device) -> dict[str, Any]:
    """Record enough runtime detail to reproduce one evaluator cell."""

    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device": str(device),
        "gpu": gpu,
    }


def json_safe(value: Any) -> Any:
    """Convert tensors, NumPy values, and non-finite floats to strict JSON."""

    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _atomic_destination(
    destination: os.PathLike[str] | str,
    *,
    mode: str,
    newline: str | None = None,
):
    """Create a same-directory temporary file for an atomic replacement."""

    path = resolve_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, tempfile.NamedTemporaryFile(
        mode=mode,
        encoding="utf-8" if "b" not in mode else None,
        newline=newline,
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )


def atomic_write_json(
    path: os.PathLike[str] | str,
    payload: Mapping[str, Any],
) -> None:
    """Atomically write deterministic, strict JSON with a trailing newline."""

    destination, temporary_handle = _atomic_destination(path, mode="w")
    temporary = Path(temporary_handle.name)
    try:
        with temporary_handle as handle:
            json.dump(
                json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(
    path: os.PathLike[str] | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    """Atomically write a schema-fixed CSV table."""

    destination, temporary_handle = _atomic_destination(path, mode="w", newline="")
    temporary = Path(temporary_handle.name)
    try:
        with temporary_handle as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: json_safe(row.get(name)) for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(payload: Any, path: os.PathLike[str] | str) -> None:
    """Atomically serialize one Torch artifact in the destination directory."""

    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def torch_load_compat(
    path: os.PathLike[str] | str,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load on both the SCNet Torch 1.12 runtime and modern Torch releases."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


__all__ = [
    "FORBIDDEN_OUTPUT_COMPONENTS",
    "assert_new_exploratory_root",
    "atomic_torch_save",
    "atomic_write_csv",
    "atomic_write_json",
    "device_provenance",
    "git_commit",
    "json_safe",
    "require_input_file",
    "reserve_exploratory_root",
    "resolve_path",
    "sha256_file",
    "source_hashes",
    "torch_load_compat",
    "verify_unchanged",
]
