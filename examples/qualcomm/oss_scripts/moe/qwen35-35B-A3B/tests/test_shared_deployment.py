from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_embedding_deployment import Device
from workflow.deployment import deploy_shared_model


def fixture(tmp_path, deploy=False):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    remote.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    groups = {"head": ["graph.qnn", "weights.bin"]}
    for group in range(10):
        groups[f"group{group}"] = ["graph.qnn", "weights.bin", "GRAPHS.json"] + [
            f"layer{layer}_{method}.io"
            for layer in range(group * 4, group * 4 + 4)
            for method in ["prefill_attention", "prefill_router", "decode"]
        ]
    for group, names in groups.items():
        (local / group).mkdir(parents=True)
        (protected / group).mkdir()
        for name in names:
            content = (group + "/" + name).encode()
            (local / group / name).write_bytes(content)
            (protected / group / name).write_bytes(content)
        (remote / group).symlink_to(protected / group, target_is_directory=True)
    return (
        SimpleNamespace(
            shared_model=local, deploy_weights=deploy, build_dir=tmp_path / "build"
        ),
        Device(),
        remote,
        protected,
    )


def test_matching_shared_directories_are_verified_without_reupload(tmp_path):
    args, device, remote, protected = fixture(tmp_path)
    before = {
        str(p.relative_to(protected)): p.read_bytes()
        for p in protected.rglob("*")
        if p.is_file()
    }
    deploy_shared_model(args, device, str(remote), "/session/prefill")
    assert all(p.is_symlink() for p in remote.iterdir())
    assert before == {
        str(p.relative_to(protected)): p.read_bytes()
        for p in protected.rglob("*")
        if p.is_file()
    }
    pushes = [c for c in device.calls if c[0] == "push"]
    assert pushes == [
        ("push", "--sync", args.build_dir / "libqwen35_shared.so", "/session/prefill/")
    ]
    assert (
        len(
            [
                c
                for c in device.calls
                if c[0] == "shell" and c[1].startswith("sha256sum ")
            ]
        )
        == 152
    )


@pytest.mark.parametrize(
    "relative",
    [
        "head/graph.qnn",
        "group0/weights.bin",
        "group3/GRAPHS.json",
        "group4/layer16_decode.io",
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_mismatched_or_missing_shared_resource_is_rejected_without_push(
    tmp_path, relative, missing
):
    args, device, remote, protected = fixture(tmp_path)
    target = protected / relative
    if missing:
        target.unlink()
    else:
        target.write_bytes(b"incorrect graph or weights")
    with pytest.raises(FileNotFoundError if missing else ValueError):
        deploy_shared_model(args, device, str(remote), "/session/prefill")
    assert not any(c[0] == "push" for c in device.calls)


def test_standard_shared_deployment_still_pushes_graph_and_weights(tmp_path):
    args, device, remote, protected = fixture(tmp_path, deploy=True)
    deploy_shared_model(args, device, str(remote), "/session/prefill")
    assert len([c for c in device.calls if c[0] == "push"]) == 12
    assert not any(
        c[0] == "shell" and c[1].startswith("sha256sum ") for c in device.calls
    )
    assert any(
        c[0] == "push" and args.shared_model / "head/weights.bin" in c
        for c in device.calls
    )
