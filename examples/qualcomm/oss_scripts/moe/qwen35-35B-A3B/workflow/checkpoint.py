"""Content-checked stages in an exclusively locked export workspace."""

import fcntl
import hashlib
import json
from pathlib import Path
import shutil


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
    def __init__(self, root, identity, filename, protected=()):
        self.root = Path(root).resolve()
        self.record = self.root / filename
        self.identity = identity
        self.lock = None
        self.dirty = False
        for source in (Path(self.root.anchor), *protected):
            source = Path(source).resolve()
            if source == self.root or source.is_relative_to(self.root):
                raise ValueError(
                    f"Workspace {self.root} contains protected input {source}"
                )

    def __enter__(self):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root.parent / ("." + self.root.name + ".export.lock")).open(
            "a"
        )
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise RuntimeError(f"An export is already using {self.root}") from error
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
                        remove_path(path)
                else:
                    remove_path(self.root)
                state = dict(identity=self.identity, stages={})
            self.root.mkdir(parents=True, exist_ok=True)
            self.state = state
            write_json(self.record, state)
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *unused):
        self.lock.close()

    def reusable(self, name, outputs):
        previous = self.state["stages"].get(name, {})
        if self.dirty or previous.get("status") != "complete":
            self.dirty = True
            return False
        try:
            valid = previous.get("outputs") == artifact_stamps(outputs)
        except (OSError, ValueError):
            valid = False
        self.dirty = not valid
        return valid

    def update(self, name, **values):
        self.state["stages"][name] = values
        write_json(self.record, self.state)
