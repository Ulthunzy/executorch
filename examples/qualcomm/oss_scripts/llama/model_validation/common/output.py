"""Publish complete artifact directories; keep interrupted attempts recoverable."""

import fcntl
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_lock(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f".{destination.name}.run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another process owns run {destination}") from error
        yield


@contextmanager
def atomic_directory(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f".{destination.name}.publish.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another process is preparing {destination}") from error
        if destination.exists():
            raise ValueError(f"Output already exists: {destination}")
        working = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.pending-", dir=destination.parent
            )
        )
        yield working
        if destination.exists():
            raise ValueError(f"Output appeared during execution: {destination}")
        working.rename(destination)
