"""Content identities and atomic JSON artifacts for resumable evaluation."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    """Hash the canonical JSON representation, rejecting nonfinite numbers."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    """Stream a SHA256 digest without loading model weights into memory."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def tree_identity(path):
    """Hash every file in a local artifact, including weight shards and configs."""
    path = Path(path).resolve()
    if path.is_file():
        files = {path.name: file_hash(path)}
    elif path.is_dir():
        files = {
            p.relative_to(path).as_posix(): file_hash(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
    else:
        raise ValueError(f"Local artifact does not exist: {path}")
    if not files:
        raise ValueError(f"Empty local artifact: {path}")
    return dict(path=str(path), files=files, sha256=digest(files))


def read_json(path):
    """Read one UTF-8 JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomically publish a complete JSON record on the same filesystem."""
    path = Path(path)
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    fd, temporary = tempfile.mkstemp(prefix=".ppl-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
