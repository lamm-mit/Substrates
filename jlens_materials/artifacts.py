# Copyright 2026.  Apache-2.0.
"""Sync a run's artifacts to / from a (private) HuggingFace **dataset** repo, so
a run produced on one machine can be continued on another.

Everything a run produces is namespaced by its ``<tag>``:

    lens_<tag>.pt            the fitted Jacobian lens (the expensive artifact)
    lens_<tag>.ckpt.pt       the fit checkpoint (optional)
    runs/<tag>.json          provenance / fixed-span ranks / metrics / exploratory candidates / swaps
    runs/<tag>_analysis.json the LLM "what we see" analysis
    runs/<tag>_swaps.json    auto_swap results (if any)
    figures/<tag>/**         every PNG/SVG + animations
    reports/<tag>_report.*   the compiled report (.tex/.pdf/.md)

``push`` uploads that set; ``pull`` restores it to the *same relative paths* on
another machine; ``list`` shows which tags a repo holds. Because the paths are
identical, after a pull you just point the normal CLIs at ``runs/<tag>.json`` or
reuse ``lens_<tag>.pt`` — no other bookkeeping.

    # after fitting the 1000-prompt lens on the Linux box:
    python artifacts.py push --tag gemma4-e4b-it-1k --repo <user>/substrates-artifacts

    # on the Mac, to continue the analysis:
    python artifacts.py pull --tag gemma4-e4b-it-1k --repo <user>/substrates-artifacts
    python analyze.py --run runs/gemma4-e4b-it-1k.json --provider openai   # continue here

    python artifacts.py list --repo <user>/substrates-artifacts            # what's stored

Auth: a HuggingFace token via ``huggingface-cli login`` (cached) or the
``HF_TOKEN`` env var. The dataset repo is created private by default.
Set ``SUBSTRATES_HF_REPO`` to avoid passing ``--repo`` every time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parent

# Latex intermediates (.aux/.log/.fls/.fdb_latexmk/.out) are regenerable — the
# report deliverables we keep are just these three.
_REPORT_EXTS = ("tex", "pdf", "md")


def _root_files(tag: str, *, include_lens: bool = True) -> list[str]:
    """Toolkit-relative paths for a tag's non-figure artifacts."""
    names = [
        f"runs/{tag}.json",
        f"runs/{tag}_analysis.json",
        f"runs/{tag}_swaps.json",
        *[f"reports/{tag}_report.{ext}" for ext in _REPORT_EXTS],
    ]
    if include_lens:
        names = [f"lens_{tag}.pt", f"lens_{tag}.ckpt.pt", *names]
    return names


def local_files(tag: str, *, include_lens: bool = True) -> list[str]:
    """Toolkit-relative paths that actually exist locally for ``tag``."""
    files = [n for n in _root_files(tag, include_lens=include_lens)
             if (TOOLKIT / n).is_file()]
    figdir = TOOLKIT / "figures" / tag
    if figdir.is_dir():
        files += [str(p.relative_to(TOOLKIT))
                  for p in sorted(figdir.rglob("*"))
                  if p.is_file() and not p.name.startswith(".")]  # skip .DS_Store etc.
    return files


def pull_patterns(tag: str, *, include_lens: bool = True) -> list[str]:
    """Glob patterns for downloading a tag (figures unknown ahead of time)."""
    pats = _root_files(tag, include_lens=include_lens)
    pats += [f"figures/{tag}/*", f"figures/{tag}/**"]   # fnmatch: both catch nested
    return pats


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}GB"


def _resolve_repo(repo: str | None) -> str:
    repo = repo or os.environ.get("SUBSTRATES_HF_REPO")
    if not repo:
        raise SystemExit("no --repo given and SUBSTRATES_HF_REPO is unset "
                         "(expected e.g. <user>/substrates-artifacts)")
    return repo


def push(tag: str, repo: str, *, private: bool = True, include_lens: bool = True,
         dry_run: bool = False) -> None:
    files = local_files(tag, include_lens=include_lens)
    if not files:
        raise SystemExit(f"no local artifacts for tag {tag!r} under {TOOLKIT} "
                         f"(nothing to push)")
    total = sum((TOOLKIT / f).stat().st_size for f in files)
    print(f"tag {tag}: {len(files)} files, {_human(total)} -> "
          f"dataset {repo} ({'private' if private else 'public'})")
    for f in files:
        print(f"  + {f}  ({_human((TOOLKIT / f).stat().st_size)})")
    if dry_run:
        print("[dry-run] nothing uploaded")
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(TOOLKIT / f))
           for f in files]
    api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                      commit_message=f"push artifacts for {tag}")
    print(f"pushed {len(files)} files for {tag} -> https://huggingface.co/datasets/{repo}")


def pull(tag: str, repo: str, *, include_lens: bool = True) -> None:
    from huggingface_hub import snapshot_download
    print(f"pulling tag {tag} from dataset {repo} -> {TOOLKIT}")
    snapshot_download(repo_id=repo, repo_type="dataset",
                      allow_patterns=pull_patterns(tag, include_lens=include_lens),
                      local_dir=str(TOOLKIT))
    got = local_files(tag, include_lens=include_lens)
    print(f"restored {len(got)} files for {tag}:")
    for f in got[:8]:
        print(f"  {f}")
    if len(got) > 8:
        print(f"  … and {len(got) - 8} more (figures)")
    print(f"\ncontinue with:  python analyze.py --run runs/{tag}.json --provider openai\n"
          f"or reuse the lens:  python run_lens.py --model <id> --lens lens_{tag}.pt --tag {tag} ...")


def list_tags(repo: str) -> None:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    tags = sorted({f[len("runs/"):-len(".json")] for f in files
                   if f.startswith("runs/") and f.endswith(".json")
                   and not f.endswith(("_analysis.json", "_swaps.json"))})
    print(f"dataset {repo}: {len(tags)} run tag(s)")
    for t in tags:
        has_lens = f"lens_{t}.pt" in files
        n_fig = sum(1 for f in files if f.startswith(f"figures/{t}/"))
        has_report = any(f.startswith(f"reports/{t}_report.") for f in files)
        print(f"  {t}  [{'lens ' if has_lens else ''}"
              f"{n_fig} figs {'report' if has_report else ''}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync run artifacts to/from a HF dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="upload a tag's artifacts")
    p.add_argument("--tag", required=True)
    p.add_argument("--repo", default=None, help="<user>/<name> (or $SUBSTRATES_HF_REPO)")
    p.add_argument("--public", action="store_true", help="create a public repo (default: private)")
    p.add_argument("--no-lens", action="store_true", help="skip the large lens_<tag>.pt")
    p.add_argument("--dry-run", action="store_true", help="list what would upload, do nothing")

    q = sub.add_parser("pull", help="download a tag's artifacts")
    q.add_argument("--tag", required=True)
    q.add_argument("--repo", default=None, help="<user>/<name> (or $SUBSTRATES_HF_REPO)")
    q.add_argument("--no-lens", action="store_true", help="skip the large lens_<tag>.pt")

    r = sub.add_parser("list", help="list tags stored in a repo")
    r.add_argument("--repo", default=None, help="<user>/<name> (or $SUBSTRATES_HF_REPO)")

    args = ap.parse_args()
    if args.cmd == "push":
        push(args.tag, _resolve_repo(args.repo), private=not args.public,
             include_lens=not args.no_lens, dry_run=args.dry_run)
    elif args.cmd == "pull":
        pull(args.tag, _resolve_repo(args.repo), include_lens=not args.no_lens)
    elif args.cmd == "list":
        list_tags(_resolve_repo(args.repo))


if __name__ == "__main__":
    main()
