#!/usr/bin/env python3
"""Run a queue of commands on a remote GPU box and ship the evidence back over Git.

The training box (a Colab A100) and the analysis side (a session without a GPU) share
no filesystem. GitHub is reachable from both, so it is the transport: this script runs
the commands of a JSON queue in order, then after every command copies the *evidence*
(reports, logs, hashes -- never weights) into a results branch and pushes it. The
analysis side fetches that branch.

Queue file::

    {
      "name": "loop-core-seed0",
      "revision": "<git sha the queue is bound to, checked before running>",
      "commands": [
        {"id": "gpu-gate", "argv": ["python", "-m", "pytest", "tests/test_gpu.py", "-q"],
         "timeout_s": 900},
        {"id": "train-gdn-s0", "argv": ["python", "scripts/run_loop_core.py", "..."],
         "timeout_s": 3600, "repeat_until": "RUN_COMPLETE", "max_attempts": 40}
      ],
      "evidence": {"roots": ["outputs/loop-core"],
                   "include": ["*.json", "*.jsonl", "*.log", "*.txt", "*.xml"],
                   "exclude": ["*.pt", "*.pth", "*.safetensors", "*.bin", "*.npy"],
                   "max_file_bytes": 50000000}
    }

Semantics, chosen after the R04 sessions (docs/17): a command that exits non-zero stops
the queue for inspection; a command with ``repeat_until`` is re-run (bounded by
``max_attempts``) until its log contains the marker, which is how one long training run
is cut into sessions that fit a Colab time limit; a completed command is never re-run,
so repeating the same queue in a new session resumes where the last one stopped.
Nothing here decides anything about the experiment: the queue orders, bounds, records
and ships.

    python scripts/colab_queue.py --queue queue.json --state STATE_DIR \
        --push github --repo speed25200-cyber/Prophet_AGI --token-env GITHUB_TOKEN
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INCLUDE = ("*.json", "*.jsonl", "*.log", "*.txt", "*.xml", "*.md", "*.csv")
DEFAULT_EXCLUDE = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    "*.npy",
    "*.npz",
    "*.gz",
    "*.zip",
    "*.tar",
    "*.ckpt",
    "*.tmp",
)
DEFAULT_MAX_FILE_BYTES = 50_000_000


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_queue(path: Path) -> dict:
    queue = json.loads(path.read_text(encoding="utf-8"))
    for key in ("name", "commands"):
        if key not in queue:
            raise ValueError(f"queue needs '{key}'")
    if not queue["name"].replace("-", "").replace("_", "").isalnum():
        raise ValueError("queue name must be alphanumeric with - or _")
    seen = set()
    for command in queue["commands"]:
        for key in ("id", "argv", "timeout_s"):
            if key not in command:
                raise ValueError(f"command needs '{key}': {command}")
        if command["id"] in seen:
            raise ValueError(f"duplicate command id {command['id']}")
        seen.add(command["id"])
        if not isinstance(command["argv"], list) or not command["argv"]:
            raise ValueError(f"argv must be a non-empty list: {command['id']}")
        if not (isinstance(command["timeout_s"], (int, float)) and command["timeout_s"] > 0):
            raise ValueError(f"timeout_s must be positive: {command['id']}")
        if "max_attempts" in command and command["max_attempts"] < 1:
            raise ValueError(f"max_attempts must be >= 1: {command['id']}")
    return queue


def current_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


class QueueState:
    """Per-command attempts and completion, persisted after every change."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"commands": {}}

    def entry(self, command_id: str) -> dict:
        return self.data["commands"].setdefault(
            command_id, {"attempts": 0, "completed": False, "exit_codes": [], "logs": []}
        )

    def save(self) -> None:
        write_json(self.path, self.data)


def run_command(command: dict, *, cwd: Path, log_path: Path, env: dict) -> int:
    """Run one attempt in its own process group; kill the whole group on timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        # The command line is in the queue file and the state; it must not be echoed
        # here, or a marker quoted in argv would count as the command's own output.
        stream.write(f"# started: {time.time()}\n")
        process = subprocess.Popen(
            command["argv"],
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=command["timeout_s"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.write(f"\n# TIMEOUT after {command['timeout_s']} s\n")
            return -1


def export_evidence(queue: dict, *, repo: Path, state_dir: Path, export_dir: Path) -> dict:
    """Copy reports and logs, never weights; record what was copied and what was skipped."""
    spec = queue.get("evidence", {})
    include = tuple(spec.get("include", DEFAULT_INCLUDE))
    exclude = tuple(spec.get("exclude", DEFAULT_EXCLUDE))
    max_bytes = int(spec.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES))
    roots = [repo / r for r in spec.get("roots", [])] + [state_dir]
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    manifest = {"queue": queue["name"], "exported_at": time.time(), "files": {}, "skipped": []}
    for root in roots:
        if not root.exists():
            continue
        label = root.name if root != state_dir else "queue-state"
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            name = path.name
            relative = Path(label) / path.relative_to(root)
            if any(fnmatch.fnmatch(name, pattern) for pattern in exclude):
                manifest["skipped"].append({"path": relative.as_posix(), "reason": "excluded"})
                continue
            if not any(fnmatch.fnmatch(name, pattern) for pattern in include):
                manifest["skipped"].append({"path": relative.as_posix(), "reason": "not included"})
                continue
            size = path.stat().st_size
            if size > max_bytes:
                manifest["skipped"].append(
                    {"path": relative.as_posix(), "reason": "too large", "bytes": size}
                )
                continue
            destination = export_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest["files"][relative.as_posix()] = {"sha256": sha256(path), "bytes": size}
    write_json(export_dir / "export-manifest.json", manifest)
    return manifest


def push_evidence(
    export_dir: Path,
    *,
    queue_name: str,
    remote: str,
    branch: str,
    work: Path,
    token: str | None,
    message: str,
) -> str:
    """Commit the export into ``results/<queue>/`` on ``branch`` and push it.

    The token is only ever placed in the remote URL of the throwaway clone under
    ``work`` and is never printed. A missing branch is created as an orphan so results
    never carry the source tree.
    """
    url = remote
    if token:
        if not remote.startswith("https://"):
            raise ValueError("a token requires an https remote")
        url = "https://x-access-token:" + token + "@" + remote[len("https://") :]
    clone = work / "results-clone"
    if clone.exists():
        shutil.rmtree(clone)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    quiet = {
        "cwd": clone,
        "env": env,
        "check": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    exists = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", url, branch],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode == 0:
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", branch, url, str(clone)],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        clone.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet"], **quiet)
        subprocess.run(["git", "checkout", "--quiet", "--orphan", branch], **quiet)
        subprocess.run(["git", "remote", "add", "origin", url], **quiet)
    subprocess.run(["git", "config", "user.email", "queue@prophet.local"], **quiet)
    subprocess.run(["git", "config", "user.name", "Prophet queue"], **quiet)
    target = clone / "results" / queue_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(export_dir, target)
    subprocess.run(["git", "add", "-A"], **quiet)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=clone, env=env)
    if changed.returncode == 0:
        return "unchanged"
    subprocess.run(["git", "commit", "--quiet", "-m", message], **quiet)
    subprocess.run(["git", "push", "--quiet", "-u", "origin", f"HEAD:{branch}"], **quiet)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()


def resolve_token(token_env: str | None) -> str | None:
    if not token_env:
        return None
    token = os.environ.get(token_env)
    if token:
        return token
    try:  # Colab keeps secrets outside the environment; read it there if available.
        from google.colab import userdata  # type: ignore

        return userdata.get(token_env)
    except Exception:
        return None


def run_queue(
    queue: dict,
    *,
    repo: Path,
    state_dir: Path,
    deadline_s: float | None,
    push: str,
    remote: str | None,
    branch: str | None,
    token_env: str | None,
    require_revision: bool = True,
) -> dict:
    began = time.time()
    if require_revision and queue.get("revision"):
        revision = current_revision(repo)
        if revision != queue["revision"]:
            raise RuntimeError(f"queue is bound to {queue['revision']}, checkout is {revision}")
    state = QueueState(state_dir / "queue-state.json")
    state.data.update({"queue": queue["name"], "revision": queue.get("revision")})
    state.save()
    env = {**os.environ, **{k: str(v) for k, v in queue.get("env", {}).items()}}
    token = resolve_token(token_env) if push == "github" else None
    export_dir = state_dir / "export"
    outcome = {"status": "complete", "stopped_on": None}

    def ship(reason: str) -> None:
        manifest = export_evidence(queue, repo=repo, state_dir=state_dir, export_dir=export_dir)
        print(
            f"EVIDENCE {len(manifest['files'])} files, {len(manifest['skipped'])} skipped",
            flush=True,
        )
        if push == "github":
            if not remote or not branch:
                raise ValueError("--push github needs --remote and --branch")
            with tempfile.TemporaryDirectory() as work:
                result = push_evidence(
                    export_dir,
                    queue_name=queue["name"],
                    remote=remote,
                    branch=branch,
                    work=Path(work),
                    token=token,
                    message=f"{queue['name']}: {reason}",
                )
            print(f"PUSHED {branch} {result}", flush=True)

    for command in queue["commands"]:
        entry = state.entry(command["id"])
        if entry["completed"]:
            print(f"SKIP {command['id']} (completed)", flush=True)
            continue
        max_attempts = int(command.get("max_attempts", 1))
        marker = command.get("repeat_until")
        while not entry["completed"] and entry["attempts"] < max_attempts:
            if deadline_s is not None and time.time() - began + command["timeout_s"] > deadline_s:
                outcome.update(status="deadline", stopped_on=command["id"])
                state.save()
                ship(f"deadline before {command['id']}")
                return outcome
            entry["attempts"] += 1
            log_path = state_dir / "logs" / f"{command['id']}-{entry['attempts']:03d}.log"
            print(f"RUN {command['id']} attempt {entry['attempts']}", flush=True)
            code = run_command(command, cwd=repo, log_path=log_path, env=env)
            entry["exit_codes"].append(code)
            entry["logs"].append(log_path.relative_to(state_dir).as_posix())
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if code == 0 and (marker is None or marker in log_text):
                entry["completed"] = True
            state.save()
            ship(f"{command['id']} attempt {entry['attempts']} exit {code}")
            if code != 0:
                outcome.update(status="failed", stopped_on=command["id"])
                return outcome
        if not entry["completed"]:
            outcome.update(status="exhausted", stopped_on=command["id"])
            return outcome
    state.save()
    ship("queue complete")
    return outcome


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True, help="persistent state directory")
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument(
        "--deadline-s",
        type=float,
        default=None,
        help="do not start a command whose timeout would exceed this budget",
    )
    ap.add_argument("--push", choices=["none", "github"], default="none")
    ap.add_argument("--remote", default=None, help="https://github.com/<owner>/<repo>.git")
    ap.add_argument("--branch", default=None, help="results branch, e.g. results/loop-core")
    ap.add_argument("--token-env", default="GITHUB_TOKEN")
    ap.add_argument("--ignore-revision", action="store_true")
    args = ap.parse_args()
    queue = load_queue(args.queue)
    args.state.mkdir(parents=True, exist_ok=True)
    outcome = run_queue(
        queue,
        repo=args.repo,
        state_dir=args.state,
        deadline_s=args.deadline_s,
        push=args.push,
        remote=args.remote,
        branch=args.branch,
        token_env=args.token_env,
        require_revision=not args.ignore_revision,
    )
    print("QUEUE", json.dumps(outcome), flush=True)
    return 0 if outcome["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
