from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lens_hub


def _write_bundle(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lens-weights")
    lens_hub.metadata_path(path).write_text(json.dumps({"format_version": 1}))


def test_hub_path_is_relative_pt_file(tmp_path):
    assert lens_hub.normalize_hub_path(None, tmp_path / "lens.pt") == "lens.pt"
    assert lens_hub.normalize_hub_path("gemma/paper/seed0.pt", "lens.pt") == (
        "gemma/paper/seed0.pt"
    )
    with pytest.raises(ValueError, match="relative file path"):
        lens_hub.normalize_hub_path("../secret.pt", "lens.pt")
    with pytest.raises(ValueError, match="end in .pt"):
        lens_hub.normalize_hub_path("gemma/lens.bin", "lens.pt")


def test_download_fetches_weights_and_metadata_as_a_pair(tmp_path, monkeypatch):
    remote_lens = tmp_path / "remote.pt"
    remote_meta = tmp_path / "remote.pt.meta.json"
    remote_lens.write_bytes(b"remote-weights")
    remote_meta.write_text(json.dumps({"format_version": 1, "corpus": {"seed": 7}}))

    requested = []

    def fake_download(**kwargs):
        requested.append(kwargs)
        return str(remote_meta if kwargs["filename"].endswith(".meta.json") else remote_lens)

    monkeypatch.setattr(lens_hub, "hf_hub_download", fake_download)
    local = tmp_path / "local" / "lens.pt"
    weights, metadata = lens_hub.download_lens_bundle(
        repo_id="owner/private-lenses",
        local_path=local,
        path_in_repo="gemma/seed7.pt",
        revision="abc123",
    )

    assert weights.read_bytes() == b"remote-weights"
    assert json.loads(metadata.read_text())["corpus"]["seed"] == 7
    assert [item["filename"] for item in requested] == [
        "gemma/seed7.pt",
        "gemma/seed7.pt.meta.json",
    ]
    assert all(item["repo_type"] == "model" for item in requested)
    assert all(item["revision"] == "abc123" for item in requested)


def test_download_reuses_complete_local_bundle(tmp_path, monkeypatch):
    local = tmp_path / "lens.pt"
    _write_bundle(local)

    def unexpected_download(**_):
        raise AssertionError("complete local bundle should not access the Hub")

    monkeypatch.setattr(lens_hub, "hf_hub_download", unexpected_download)
    assert lens_hub.download_lens_bundle(
        repo_id="owner/private-lenses", local_path=local
    ) == (local, lens_hub.metadata_path(local))


def test_upload_commits_weights_and_metadata_and_records_location(tmp_path, monkeypatch):
    local = tmp_path / "lens.pt"
    _write_bundle(local)

    class FakeApi:
        def __init__(self):
            self.created = None
            self.commit = None

        def create_repo(self, **kwargs):
            self.created = kwargs

        def repo_info(self, **_):
            return SimpleNamespace(private=True)

        def create_commit(self, **kwargs):
            self.commit = kwargs
            return SimpleNamespace(commit_url="https://huggingface.co/owner/private/commit/123")

    api = FakeApi()
    monkeypatch.setattr(lens_hub, "HfApi", lambda: api)
    url = lens_hub.upload_lens_bundle(
        repo_id="owner/private",
        local_path=local,
        path_in_repo="gemma/paper/seed0.pt",
        private=True,
    )

    assert url.endswith("/commit/123")
    assert api.created["private"] is True
    assert api.created["repo_type"] == "model"
    assert [operation.path_in_repo for operation in api.commit["operations"]] == [
        "gemma/paper/seed0.pt",
        "gemma/paper/seed0.pt.meta.json",
    ]
    metadata = json.loads(lens_hub.metadata_path(local).read_text())
    assert metadata["hub"]["repo_id"] == "owner/private"
    assert metadata["hub"]["path_in_repo"] == "gemma/paper/seed0.pt"


def test_private_upload_refuses_existing_public_repo(tmp_path, monkeypatch):
    local = tmp_path / "lens.pt"
    _write_bundle(local)

    class PublicApi:
        def create_repo(self, **_):
            return None

        def repo_info(self, **_):
            return SimpleNamespace(private=False)

    monkeypatch.setattr(lens_hub, "HfApi", PublicApi)
    with pytest.raises(ValueError, match="refusing to upload lens to public"):
        lens_hub.upload_lens_bundle(
            repo_id="owner/public", local_path=local, private=True
        )
