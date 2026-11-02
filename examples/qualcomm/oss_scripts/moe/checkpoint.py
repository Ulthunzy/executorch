"""Content-checked stages in an exclusively locked export workspace."""

import fcntl
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys


def input_stamp(path):
    path = Path(path)
    stat = path.stat()
    stamp = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if path.is_file() and stat.st_size <= 1024 * 1024:
        stamp["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return stamp


@contextmanager
def workspace_lock(root):
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ("." + root.name + ".export.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"An export is already using {root}") from error
        yield lock


def artifact_stamps(paths):
    stamps = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty artifact: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stamps[str(path)] = dict(bytes=path.stat().st_size, sha256=digest)
    return stamps


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


class StageCheckpoint:
    def __init__(self, root, identity, filename, protected=(), preserved=()):
        self.root = Path(root).resolve()
        self.record = self.root / filename
        self.identity = identity
        self.lock = None
        self.dirty = False
        self.preserved = {Path(path).absolute() for path in preserved}
        if any(path.parent != self.root for path in self.preserved):
            raise ValueError("Preserved caches must be immediate workspace children")
        self.preserved.update(
            self.root / f".{path.name}.export.lock" for path in tuple(self.preserved)
        )
        for source in (Path(self.root.anchor), *protected):
            source = Path(source).resolve()
            cached = any(source.is_relative_to(path) for path in self.preserved)
            if source == self.root or (source.is_relative_to(self.root) and not cached):
                raise ValueError(
                    f"Workspace {self.root} contains protected input {source}"
                )

    def __enter__(self):
        self.lock_context = workspace_lock(self.root)
        self.lock = self.lock_context.__enter__()
        try:
            try:
                state = json.loads(self.record.read_text())
            except (OSError, ValueError):
                state = None
            if (
                not isinstance(state, dict)
                or state.get("identity") != self.identity
                or not isinstance(state.get("stages"), dict)
            ):
                if self.root.is_dir():
                    for path in self.root.iterdir():
                        if path not in self.preserved:
                            remove_path(path)
                else:
                    remove_path(self.root)
                state = dict(identity=self.identity, stages={})
            self.root.mkdir(parents=True, exist_ok=True)
            self.state = state
            write_json(self.record, state)
            return self
        except BaseException:
            self.lock_context.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *unused):
        self.lock_context.__exit__(*unused)

    def reusable(self, name, outputs):
        if self.dirty:
            return False
        valid = self.valid(name, outputs)
        self.dirty = not valid
        return valid

    def valid(self, name, outputs):
        previous = self.state["stages"].get(name, {})
        if not isinstance(previous, dict) or previous.get("status") != "complete":
            return False
        try:
            valid = previous.get("outputs") == artifact_stamps(outputs)
        except (OSError, ValueError):
            valid = False
        return valid

    def update(self, name, **values):
        self.state["stages"][name] = values
        write_json(self.record, self.state)
