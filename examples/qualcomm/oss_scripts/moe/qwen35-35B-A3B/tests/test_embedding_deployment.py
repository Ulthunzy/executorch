import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow.deployment import deploy_embedding


class Device:
    def __init__(self):
        self.calls = []

    def adb(self, *args):
        self.calls.append(args)
        if args[0] == "shell" and args[1].startswith("sha256sum "):
            target = Path(shlex.split(args[1])[1])
            if not target.is_file():
                raise subprocess.CalledProcessError(1, ["sha256sum", str(target)])
            return (
                hashlib.sha256(target.read_bytes()).hexdigest()
                + "  "
                + str(target)
                + "\n"
            )
        return ""


def fixture(tmp_path, *, content=b"selected embedding", exists=True, deploy=False):
    local = tmp_path / "embedding.int4"
    local.write_bytes(b"selected embedding")
    remote = tmp_path / "remote"
    remote.mkdir()
    underlying = tmp_path / "existing-embedding.bin"
    if exists:
        underlying.write_bytes(content)
    (remote / "embedding.int4").symlink_to(underlying)
    args = SimpleNamespace(
        embedding_table=local,
        artifacts=tmp_path,
        remote_root=str(remote),
        deploy_weights=deploy,
    )
    return args, Device(), remote / "embedding.int4", underlying


def test_matching_remote_embedding_symlink_is_verified_without_push(tmp_path):
    args, device, link, original = fixture(tmp_path)
    oldlink = link.readlink()
    oldbytes = original.read_bytes()
    deploy_embedding(args, device, "/session/prefill", "/session/decode")
    assert (
        link.is_symlink()
        and link.readlink() == oldlink
        and original.read_bytes() == oldbytes
    )
    assert not any(call[0] == "push" for call in device.calls)
    assert (
        len(
            [
                call
                for call in device.calls
                if call[0] == "shell" and call[1].startswith("ln -sfn ")
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "exists,content", [(False, b""), (True, b"different embedding")]
)
def test_missing_or_mismatched_embedding_never_overwrites_link_target(
    tmp_path, exists, content
):
    args, device, link, original = fixture(tmp_path, exists=exists, content=content)
    error = FileNotFoundError if not exists else ValueError
    with pytest.raises(error):
        deploy_embedding(args, device, "/session/prefill", "/session/decode")
    assert link.is_symlink() and len(device.calls) == 1
    assert not original.exists() if not exists else original.read_bytes() == content


def test_normal_deployment_keeps_original_sync_push(tmp_path):
    args, device, link, original = fixture(tmp_path, deploy=True)
    deploy_embedding(args, device, "/session/prefill", "/session/decode")
    assert device.calls[0] == ("push", "--sync", args.embedding_table, str(link))
