# %% [markdown]
# # Prophet loop-core runner (Colab A100)
#
# Runs one queue of commands from `queue/loop_core/*.json` at a pinned revision and ships
# the evidence (reports, logs, hashes; never weights) to the `results/loop-core` branch on
# GitHub after every command. Checkpoints stay on Drive. Repeat the queue cell in a new
# session to resume: completed commands are skipped.
#
# One-time setup on your side: a fine-grained GitHub token with *Contents: read and write*
# on `speed25200-cyber/Prophet_AGI`, stored in Colab secrets (key icon, left bar) under the
# name `GITHUB_TOKEN`, with notebook access enabled. Without it the queue still runs and the
# evidence stays under `STATE` on Drive.
#
# This file is the notebook's source (`jupytext`-style `# %%` cells). Open it in Colab via
# *File > Upload notebook* after converting, or paste the cells one by one.

# %%
# Cell 1 -- Drive, pinned checkout, dependencies, GPU gate.
from google.colab import drive  # type: ignore

drive.mount("/content/drive")

import os, subprocess, sys  # noqa: E401
from pathlib import Path

BRANCH = "claude/codex-results-analysis-5jz95y"
REVISION = None  # set to the exact commit the queue file names; None = branch head
QUEUE = "queue/loop_core/seed0.json"

PERSISTENT = Path("/content/drive/MyDrive/Prophet_AGI/loop-core")
STATE = PERSISTENT / "queue-state"
PERSISTENT.mkdir(parents=True, exist_ok=True)
repo = Path("/content/Prophet_AGI")
if not repo.exists():
    subprocess.run(["git", "clone", "--quiet", "https://github.com/speed25200-cyber/Prophet_AGI.git",
                    str(repo)], check=True)
subprocess.run(["git", "fetch", "--quiet", "origin", BRANCH], cwd=repo, check=True)
subprocess.run(["git", "checkout", "--quiet", REVISION or f"origin/{BRANCH}"], cwd=repo, check=True)
print("checkout", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())

os.environ["TRITON_F32_DEFAULT"] = "tf32x3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo) + "[dev,gpu]",
                "datasets==5.0.1", "huggingface-hub==1.32.0"], check=True)
gate = subprocess.run([sys.executable, "-m", "pytest", "tests/test_gpu.py", "-q", "--tb=short"],
                      cwd=repo, capture_output=True, text=True, timeout=900)
print(gate.stdout[-4000:], gate.stderr[-2000:])
gate.check_returncode()

# %%
# Cell 2 -- Run the queue. Re-run this cell in a new session to resume.
import subprocess, sys  # noqa: E401

log = open("/content/queue.log", "a", buffering=1)
queue_process = subprocess.Popen(
    [sys.executable, "-u", "scripts/colab_queue.py", "--queue", QUEUE, "--state", str(STATE),
     "--deadline-s", str(11 * 3600), "--push", "github",
     "--remote", "https://github.com/speed25200-cyber/Prophet_AGI.git",
     "--branch", "results/loop-core", "--token-env", "GITHUB_TOKEN"],
    cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    env={**os.environ, "PROPHET_PERSISTENT": str(PERSISTENT)})
print("queue pid", queue_process.pid)

# %%
# Cell 3 -- Watch. Run as often as you like; it never changes anything.
print("exit code:", queue_process.poll())
print(open("/content/queue.log").read()[-6000:])

# %%
# Cell 4 -- Only after the queue reports QUEUE {"status": ...}: flush Drive, release the GPU.
assert queue_process.poll() is not None, "queue still running"
from google.colab import drive, runtime  # type: ignore

drive.flush_and_unmount()
runtime.unassign()
