"""
Shared helpers for writing experiment results.

Every experiment script writes one JSON record with:
    - the script arguments
    - the measured results
    - run metadata (versions, git commit, CPU binding, SLURM job)

so that numbers stay comparable across iterations.
"""

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess


# src/sparse_inference/results.py -> repository root
REPO_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

SPARSEGPT_FILE = os.path.join(
    REPO_DIR,
    "third_party",
    "sparsegpt",
    "sparsegpt.py",
)


def _git(*args):
    try:
        return subprocess.run(
            ["git", "-C", REPO_DIR, *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _sha256(path):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return platform.processor()


def run_metadata():
    import torch

    meta = {
        "timestamp": datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "sparsegpt_sha256": _sha256(SPARSEGPT_FILE),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "sched_affinity": sorted(os.sched_getaffinity(0)),
    }

    for package in ["transformers", "datasets"]:
        try:
            meta[f"{package}_version"] = __import__(package).__version__
        except ImportError:
            meta[f"{package}_version"] = None

    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "CPU_BIND",
        "MEM_BIND",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NODELIST",
    ]:
        meta[var.lower()] = os.environ.get(var)

    return meta


def model_revision(model):
    """
    Hugging Face commit hash of the loaded weights (None for local dirs
    that were not downloaded from the Hub).
    """
    return getattr(model.config, "_commit_hash", None)


def write_json(path, record):
    """
    Write a result record. Metadata is added automatically.
    """
    if path is None:
        return

    path = os.path.expandvars(os.path.expanduser(path))

    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )

    record = dict(record)
    record.setdefault("meta", run_metadata())

    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"Results written to {path}")


def read_json(path):
    with open(path) as f:
        return json.load(f)
