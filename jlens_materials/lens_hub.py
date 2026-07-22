# Copyright 2026. Apache-2.0.
"""Private Hugging Face Hub storage for fitted Jacobian-lens bundles.

A usable lens artifact is a pair: ``weights.pt`` and
``weights.pt.meta.json``.  Uploads commit both files atomically, and downloads
stage both files before replacing the local pair.  Authentication is delegated
to ``HF_TOKEN`` or the token saved by ``hf auth login``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


def metadata_path(lens_path: str | Path) -> Path:
    return Path(f"{lens_path}.meta.json")


def normalize_hub_path(path_in_repo: str | None, local_path: str | Path) -> str:
    """Return a safe relative POSIX path for a Hub model repository."""
    candidate = path_in_repo or Path(local_path).name
    path = PurePosixPath(candidate)
    if not candidate or path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(
            "--hf-lens-path must be a relative file path inside the Hub repository"
        )
    if path.suffix != ".pt":
        raise ValueError("--hf-lens-path must end in .pt")
    return path.as_posix()


def _atomic_copy(source: str | Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download_lens_bundle(
    *,
    repo_id: str,
    local_path: str | Path,
    path_in_repo: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download a lens and provenance sidecar from a Hub model repository."""
    local = Path(local_path)
    local_meta = metadata_path(local)
    if not force and local.is_file() and local_meta.is_file():
        return local, local_meta

    remote = normalize_hub_path(path_in_repo, local)
    remote_meta = f"{remote}.meta.json"
    downloaded_lens = hf_hub_download(
        repo_id=repo_id,
        filename=remote,
        repo_type="model",
        revision=revision,
    )
    downloaded_meta = hf_hub_download(
        repo_id=repo_id,
        filename=remote_meta,
        repo_type="model",
        revision=revision,
    )

    # Parse before replacing either local file so a broken sidecar cannot
    # silently downgrade a paper lens to an unverified artifact.
    json.loads(Path(downloaded_meta).read_text(encoding="utf-8"))
    _atomic_copy(downloaded_lens, local)
    _atomic_copy(downloaded_meta, local_meta)
    return local, local_meta


def upload_lens_bundle(
    *,
    repo_id: str,
    local_path: str | Path,
    path_in_repo: str | None = None,
    revision: str | None = "main",
    private: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload a lens pair in one commit and return the commit URL or id."""
    local = Path(local_path)
    local_meta = metadata_path(local)
    if not local.is_file():
        raise FileNotFoundError(f"lens weights not found: {local}")
    if not local_meta.is_file():
        raise FileNotFoundError(f"lens provenance sidecar not found: {local_meta}")

    remote = normalize_hub_path(path_in_repo, local)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
    if private and not bool(getattr(repo_info, "private", False)):
        raise ValueError(
            f"refusing to upload lens to public repository {repo_id!r}; "
            "make it private or explicitly pass --no-hf-private"
        )

    metadata = json.loads(local_meta.read_text(encoding="utf-8"))
    metadata["hub"] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "path_in_repo": remote,
        "revision": revision or "main",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision or "main",
        commit_message=commit_message or f"Upload Jacobian lens {remote}",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local),
            CommitOperationAdd(
                path_in_repo=f"{remote}.meta.json", path_or_fileobj=metadata_bytes
            ),
        ],
    )
    # Only claim a Hub location locally after the remote commit succeeds.
    local_meta.write_bytes(metadata_bytes)
    return str(getattr(info, "commit_url", None) or getattr(info, "oid", info))
