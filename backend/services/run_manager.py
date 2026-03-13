"""Async run manager for long-running model training tasks."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class RunInfo:
    run_id: str
    status: RunStatus = RunStatus.PENDING
    progress: int = 0
    est_seconds: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)


_runs: dict[str, RunInfo] = {}
MAX_RUNS = 50


def create_run(est_seconds: int = 30) -> RunInfo:
    if len(_runs) >= MAX_RUNS:
        oldest = sorted(_runs, key=lambda k: _runs[k].created_at)[: len(_runs) - MAX_RUNS + 1]
        for k in oldest:
            del _runs[k]
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run = RunInfo(run_id=run_id, status=RunStatus.TRAINING, est_seconds=est_seconds)
    _runs[run_id] = run
    return run


def get_run(run_id: str) -> RunInfo | None:
    return _runs.get(run_id)


def update_run(run_id: str, **kwargs) -> None:
    run = _runs.get(run_id)
    if run:
        for k, v in kwargs.items():
            setattr(run, k, v)


def complete_run(run_id: str, metrics: dict, results: dict) -> None:
    update_run(run_id, status=RunStatus.COMPLETE, progress=100,
               metrics=metrics, results=results)


def fail_run(run_id: str, error: str) -> None:
    update_run(run_id, status=RunStatus.FAILED, error=error)


def list_runs() -> list[RunInfo]:
    return list(_runs.values())
