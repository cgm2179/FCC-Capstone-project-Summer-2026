"""workspace_store.py — versioned, git-backed store for SAVEd projects.

Saved projects live on a dedicated orphan branch **workspace-store** (NEVER main), checked out
via a git worktree at ``.workspace-store/`` (git-ignored) so the user's working tree is never
touched. Each SAVE is a commit = a version, newest on top. ``restore`` brings back an older
version. On startup the backend ``materialize``s the branch into the live store, so a fresh
clone "has" the saved workspaces + their full history without bloating main.

The orphan branch is created with an empty root commit via ``git commit-tree`` (plumbing), which
does not touch or check out anything in the user's working tree. Pushing to the remote is a
separate, explicit step (``push``) — SAVE commits locally; the caller decides when to publish.
"""
from __future__ import annotations

import datetime
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH = "workspace-store"
WORKTREE = REPO_ROOT / ".workspace-store"
PROJECTS_DIR = Path(__file__).resolve().parent / "projects"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"   # git's well-known empty tree object
_IGNORE = shutil.ignore_patterns("_tmp", "__pycache__", "*.pyc")


class StoreError(RuntimeError):
    pass


def _git(args, cwd=REPO_ROOT, check=True) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *[str(a) for a in args]], cwd=str(cwd),
                       capture_output=True, text=True)
    if check and p.returncode != 0:
        raise StoreError(f"git {' '.join(str(a) for a in args)} failed:\n{p.stderr.strip()}")
    return p


def _branch_exists() -> str | None:
    if _git(["rev-parse", "--verify", "--quiet", BRANCH], check=False).returncode == 0:
        return "local"
    if _git(["rev-parse", "--verify", "--quiet", f"origin/{BRANCH}"], check=False).returncode == 0:
        return "remote"
    return None


def ensure_worktree() -> Path:
    """Create/attach the .workspace-store worktree bound to the workspace-store branch."""
    if (WORKTREE / ".git").exists():
        return WORKTREE
    if _branch_exists() is None:
        # Empty orphan root commit via plumbing — does NOT touch the user's working tree.
        p = subprocess.run(["git", "commit-tree", EMPTY_TREE, "-m", "init workspace-store"],
                           cwd=str(REPO_ROOT), capture_output=True, text=True)
        if p.returncode != 0:
            raise StoreError("could not create workspace-store branch:\n" + p.stderr.strip())
        _git(["branch", BRANCH, p.stdout.strip()])
    _git(["worktree", "add", str(WORKTREE), BRANCH])   # DWIMs a local branch from origin/ if needed
    return WORKTREE


def list_versions(pid: str) -> list[dict]:
    if _branch_exists() is None:
        return []
    ensure_worktree()
    p = _git(["log", "--format=%H%x09%cI%x09%s", "--", pid], cwd=WORKTREE, check=False)
    out = []
    for line in p.stdout.strip().splitlines():
        if "\t" not in line:
            continue
        sha, when, subj = line.split("\t", 2)
        out.append({"sha": sha, "time": when, "subject": subj})
    return out   # newest first (git log order)


def save_version(pid: str) -> dict | None:
    """Copy the live project into the worktree and commit it as a new version. Returns the new
    version info, or None if nothing changed since the last save."""
    src = PROJECTS_DIR / pid
    if not src.is_dir():
        raise StoreError(f"nothing to save for '{pid}' (built-in projects are already committed)")
    ensure_worktree()
    dst = WORKTREE / pid
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_IGNORE)
    _git(["add", "-A", pid], cwd=WORKTREE)
    if _git(["diff", "--cached", "--quiet"], cwd=WORKTREE, check=False).returncode == 0:
        return None   # identical to the last saved version
    n = len(list_versions(pid)) + 1
    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    _git(["commit", "-m", f"save: {pid} v{n} {ts}"], cwd=WORKTREE)
    sha = _git(["rev-parse", "HEAD"], cwd=WORKTREE).stdout.strip()
    return {"sha": sha, "version": n, "time": ts}


def restore(pid: str, sha: str) -> None:
    """Check an older version of a project out of the branch and back into the live store."""
    ensure_worktree()
    _git(["checkout", sha, "--", pid], cwd=WORKTREE)
    dst = PROJECTS_DIR / pid
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(WORKTREE / pid, dst)
    _git(["reset", "--hard", "HEAD"], cwd=WORKTREE, check=False)   # unstage the historical checkout


def materialize() -> int:
    """On startup: copy every saved project from the branch into the live store (skip ones that
    already exist locally). Returns how many were materialized. Best-effort — never fatal."""
    try:
        if _branch_exists() is None:
            return 0
        ensure_worktree()
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        n = 0
        for item in WORKTREE.iterdir():
            if item.name in (".git", "README.md") or not item.is_dir():
                continue
            dst = PROJECTS_DIR / item.name
            if not dst.exists():
                shutil.copytree(item, dst, ignore=_IGNORE)
                n += 1
        return n
    except StoreError as e:
        print(f"[workspace-store] materialize skipped: {e}")
        return 0


def push() -> None:
    """Publish the workspace-store branch to origin. Outward-facing — call only on explicit intent."""
    ensure_worktree()
    _git(["push", "-u", "origin", BRANCH], cwd=WORKTREE)
