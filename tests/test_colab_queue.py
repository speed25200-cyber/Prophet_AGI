"""The queue orders, bounds, records and ships; it never decides.

All cases run real subprocesses on the CPU and a local bare Git repository as the
remote, so the push path is exercised without network or token.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.colab_queue import export_evidence, load_queue, push_evidence, run_queue

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git required")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=repo, check=True)
    return repo


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _queue(repo: Path, commands: list[dict], name: str = "test-queue") -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return load_queue_dict(
        {
            "name": name,
            "revision": revision,
            "commands": commands,
            "evidence": {"roots": ["outputs"]},
        }
    )


def load_queue_dict(data: dict) -> dict:
    path = Path.cwd() / "_queue.json"  # load_queue reads a file; write then delete
    try:
        path.write_text(json.dumps(data))
        return load_queue(path)
    finally:
        path.unlink(missing_ok=True)


def test_queue_completes_repeats_until_marker_and_skips_completed_commands(tmp_path):
    repo = _repo(tmp_path)
    state = tmp_path / "state"
    counter = repo / "counter.txt"
    commands = [
        {"id": "hello", "argv": _py("print('hello')"), "timeout_s": 30},
        {
            "id": "sessions",
            "timeout_s": 30,
            "repeat_until": "RUN_COMPLETE",
            "max_attempts": 5,
            "argv": _py(
                "from pathlib import Path; p=Path('counter.txt'); n=int(p.read_text() or 0) "
                "if p.exists() else 0; n+=1; p.write_text(str(n)); "
                "print('RUN_COMPLETE' if n==3 else 'SESSION_COMPLETE_RESUME_REQUIRED')"
            ),
        },
        {
            "id": "report",
            "timeout_s": 30,
            "argv": _py(
                "from pathlib import Path; d=Path('outputs/run'); d.mkdir(parents=True, exist_ok=True); "
                "(d/'report.json').write_text('{\"loss\": 1.0}'); (d/'ckpt_slot0.pt').write_bytes(b'w'*10)"
            ),
        },
    ]
    queue = _queue(repo, commands)
    outcome = run_queue(
        queue,
        repo=repo,
        state_dir=state,
        deadline_s=None,
        push="none",
        remote=None,
        branch=None,
        token_env=None,
    )
    assert outcome == {"status": "complete", "stopped_on": None}
    assert counter.read_text() == "3"
    recorded = json.loads((state / "queue-state.json").read_text())["commands"]
    assert recorded["sessions"]["attempts"] == 3 and recorded["sessions"]["completed"]
    assert recorded["sessions"]["exit_codes"] == [0, 0, 0]
    manifest = json.loads((state / "export" / "export-manifest.json").read_text())
    assert "outputs/run/report.json" in manifest["files"]
    assert all("ckpt_slot0.pt" not in f for f in manifest["files"])
    assert any(
        s["path"].endswith("ckpt_slot0.pt") and s["reason"] == "excluded"
        for s in manifest["skipped"]
    )
    assert any(f.startswith("queue-state/logs/sessions-002.log") for f in manifest["files"])

    # A second session with the same queue resumes: nothing re-runs.
    outcome = run_queue(
        queue,
        repo=repo,
        state_dir=state,
        deadline_s=None,
        push="none",
        remote=None,
        branch=None,
        token_env=None,
    )
    assert outcome["status"] == "complete"
    assert counter.read_text() == "3"
    assert (
        json.loads((state / "queue-state.json").read_text())["commands"]["hello"]["attempts"] == 1
    )


def test_a_failing_command_stops_the_queue_and_later_commands_never_run(tmp_path):
    repo = _repo(tmp_path)
    marker = repo / "never.txt"
    commands = [
        {"id": "boom", "argv": _py("import sys; print('x'); sys.exit(3)"), "timeout_s": 30},
        {"id": "after", "argv": _py("open('never.txt','w').write('ran')"), "timeout_s": 30},
    ]
    outcome = run_queue(
        _queue(repo, commands),
        repo=repo,
        state_dir=tmp_path / "s",
        deadline_s=None,
        push="none",
        remote=None,
        branch=None,
        token_env=None,
    )
    assert outcome == {"status": "failed", "stopped_on": "boom"}
    assert not marker.exists()
    recorded = json.loads((tmp_path / "s" / "queue-state.json").read_text())["commands"]
    assert recorded["boom"]["exit_codes"] == [3] and not recorded["boom"]["completed"]
    assert "after" not in recorded


def test_timeout_kills_the_process_and_counts_as_failure(tmp_path):
    repo = _repo(tmp_path)
    commands = [
        {
            "id": "slow",
            "argv": _py("import time; print('start', flush=True); time.sleep(30)"),
            "timeout_s": 1,
        }
    ]
    outcome = run_queue(
        _queue(repo, commands),
        repo=repo,
        state_dir=tmp_path / "s",
        deadline_s=None,
        push="none",
        remote=None,
        branch=None,
        token_env=None,
    )
    assert outcome["status"] == "failed"
    recorded = json.loads((tmp_path / "s" / "queue-state.json").read_text())["commands"]["slow"]
    assert recorded["exit_codes"] == [-1]
    assert "TIMEOUT" in (tmp_path / "s" / "logs" / "slow-001.log").read_text()


def test_exhausted_attempts_without_marker_stop_the_queue(tmp_path):
    repo = _repo(tmp_path)
    commands = [
        {
            "id": "never-done",
            "argv": _py("print('SESSION_COMPLETE')"),
            "timeout_s": 30,
            "repeat_until": "RUN_COMPLETE",
            "max_attempts": 2,
        }
    ]
    outcome = run_queue(
        _queue(repo, commands),
        repo=repo,
        state_dir=tmp_path / "s",
        deadline_s=None,
        push="none",
        remote=None,
        branch=None,
        token_env=None,
    )
    assert outcome == {"status": "exhausted", "stopped_on": "never-done"}


def test_deadline_stops_before_a_command_that_cannot_fit(tmp_path):
    repo = _repo(tmp_path)
    commands = [{"id": "long", "argv": _py("print('x')"), "timeout_s": 3600}]
    outcome = run_queue(
        _queue(repo, commands),
        repo=repo,
        state_dir=tmp_path / "s",
        deadline_s=10,
        push="none",
        remote=None,
        branch=None,
        token_env=None,
    )
    assert outcome == {"status": "deadline", "stopped_on": "long"}


def test_queue_bound_to_another_revision_is_refused(tmp_path):
    repo = _repo(tmp_path)
    queue = _queue(repo, [{"id": "a", "argv": _py("print(1)"), "timeout_s": 5}])
    queue["revision"] = "0" * 40
    with pytest.raises(RuntimeError, match="bound to"):
        run_queue(
            queue,
            repo=repo,
            state_dir=tmp_path / "s",
            deadline_s=None,
            push="none",
            remote=None,
            branch=None,
            token_env=None,
        )


def test_malformed_queues_are_rejected():
    with pytest.raises(ValueError, match="needs 'commands'"):
        load_queue_dict({"name": "x"})
    with pytest.raises(ValueError, match="duplicate"):
        load_queue_dict(
            {
                "name": "x",
                "commands": [
                    {"id": "a", "argv": ["true"], "timeout_s": 1},
                    {"id": "a", "argv": ["true"], "timeout_s": 1},
                ],
            }
        )
    with pytest.raises(ValueError, match="timeout_s"):
        load_queue_dict({"name": "x", "commands": [{"id": "a", "argv": ["true"], "timeout_s": 0}]})
    with pytest.raises(ValueError, match="alphanumeric"):
        load_queue_dict({"name": "x y", "commands": []})


def test_evidence_is_pushed_to_a_results_branch_and_unchanged_content_is_not_recommitted(tmp_path):
    repo = _repo(tmp_path)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(bare)], check=True)
    state = tmp_path / "state"
    commands = [
        {
            "id": "report",
            "timeout_s": 30,
            "argv": _py(
                "from pathlib import Path; d=Path('outputs/run'); d.mkdir(parents=True, exist_ok=True); "
                "(d/'evaluation.json').write_text('{\"bpb\": 1.2}'); (d/'weights.safetensors').write_bytes(b'0'*8)"
            ),
        }
    ]
    queue = _queue(repo, commands, name="lc-seed0")
    outcome = run_queue(
        queue,
        repo=repo,
        state_dir=state,
        deadline_s=None,
        push="github",
        remote=bare.as_uri(),
        branch="results/lc",
        token_env=None,
    )
    assert outcome["status"] == "complete"
    listing = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "results/lc"], cwd=bare, text=True
    ).split()
    assert "results/lc-seed0/outputs/run/evaluation.json" in listing
    assert "results/lc-seed0/export-manifest.json" in listing
    assert not any(name.endswith("weights.safetensors") for name in listing)
    assert not any(name == "README" for name in listing), "results branch carries no source tree"
    export = state / "export"
    again = push_evidence(
        export,
        queue_name="lc-seed0",
        remote=bare.as_uri(),
        branch="results/lc",
        work=tmp_path / "w",
        token=None,
        message="again",
    )
    assert again == "unchanged"
    with pytest.raises(ValueError, match="https remote"):
        push_evidence(
            export,
            queue_name="lc-seed0",
            remote=bare.as_uri(),
            branch="results/lc",
            work=tmp_path / "w2",
            token="secret",
            message="x",
        )


def test_export_skips_oversized_files_and_records_why(tmp_path):
    repo = _repo(tmp_path)
    big = repo / "outputs" / "big.log"
    big.parent.mkdir()
    big.write_bytes(b"x" * 2048)
    (repo / "outputs" / "small.json").write_text("{}")
    queue = {
        "name": "q",
        "commands": [],
        "evidence": {"roots": ["outputs"], "max_file_bytes": 1024},
    }
    manifest = export_evidence(
        queue, repo=repo, state_dir=tmp_path / "s", export_dir=tmp_path / "e"
    )
    assert list(manifest["files"]) == ["outputs/small.json"]
    assert manifest["skipped"] == [
        {"path": "outputs/big.log", "reason": "too large", "bytes": 2048}
    ]


def test_cli_runs_a_queue_file_and_reports_status(tmp_path):
    repo = _repo(tmp_path)
    queue = _queue(repo, [{"id": "a", "argv": _py("print('ok')"), "timeout_s": 10}])
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue))
    script = Path(__file__).resolve().parent.parent / "scripts" / "colab_queue.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--queue",
            str(path),
            "--state",
            str(tmp_path / "s"),
            "--repo",
            str(repo),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "complete"' in result.stdout
