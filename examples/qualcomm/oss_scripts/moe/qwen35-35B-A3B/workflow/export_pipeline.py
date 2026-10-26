"""Serial export stages, configuration snapshots and restart checkpoints."""

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .export_config import REPO, ROOT, ExportConfig


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    cleanup: tuple[Path, ...] = ()


def file_stamp(path):
    stat = path.stat()
    stamp = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if path.is_file() and stat.st_size <= 1024 * 1024:
        stamp["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return stamp


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class ExportPipeline:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.settings = config.model_settings
        self.stages = self.plan()

    def plan(self):
        c = self.config
        out = c.output
        snapshot = out / "MODEL_CONFIG.json"
        common = ("--model-config", str(snapshot))
        binding = ("--qnn-binding", str(c.qnn_binding)) if c.qnn_binding else ()
        stages = []

        def stage(name, script, arguments, outputs, cleanup=()):
            command = (sys.executable, "-u", str(ROOT / script), *map(str, arguments))
            stages.append(Stage(name, command, tuple(outputs), tuple(cleanup)))

        if c.build and "text" in c.components:
            stages.append(
                Stage(
                    "build",
                    ("bash", str(ROOT / "build.sh")),
                    tuple(
                        c.build_dir / name
                        for name in (
                            "qwen35_moe_runner",
                            "prefill_handoff_runner",
                            "libmoe_decode.so",
                            "libqwen35_shared.so",
                            "capture_weights",
                            "TARGET.json",
                            "op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so",
                            f"op-package/hexagon-{c.dsp_arch}/libQnnDeltaNetOpPackage_HTP.so",
                            "native-prefill/libinteger_rpc_skel.so",
                            "native-decode/libinteger_rpc_skel.so",
                        )
                    ),
                )
            )
        text = "text" in c.components
        if text and c.source == "safetensors":
            stage(
                "quantize",
                "quantization/prepare_model.py",
                (
                    *common,
                    "--model",
                    c.model,
                    "--output",
                    c.calibration_dir,
                    "--corpus",
                    c.corpus,
                    "--expert-batch",
                    c.expert_batch,
                    *binding,
                ),
                (
                    c.calibration_dir / "QUANTIZATION_COMPLETE.json",
                    c.calibration_dir / "cold-experts.i32",
                    c.calibration_dir / "embedding.int4",
                    *(
                        c.calibration_dir / f"banks/layer{layer}/{name}"
                        for layer in range(40)
                        for name in ("BANK.json", "experts.n2")
                    ),
                    *(
                        c.calibration_dir / f"indices/layer{layer}/bank-index.bin"
                        for layer in range(40)
                    ),
                ),
            )
        if text:
            stage(
                "head",
                "export/head.py",
                (
                    *common,
                    "--source",
                    c.calibration_dir / "head/head-qdq.pt2",
                    "--output",
                    out / "head",
                    *binding,
                ),
                (out / "head/HEAD.json", out / "head/head.bin"),
                (out / "head/head.pte",),
            )
            for group in range(10):
                directory = out / f"qnn-shared/group{group}"
                layers = range(group * 4, group * 4 + 4)
                router = (
                    ("--router-qdq", c.calibration_dir / "router")
                    if self.settings.quantization.router_shared == "w8a16_gptq"
                    else ()
                )
                stage(
                    f"qnn-group{group}",
                    "export/shared.py",
                    (
                        *common,
                        "--prefill-source",
                        c.calibration_dir / "layers",
                        "--decode-source",
                        c.calibration_dir / "layers",
                        *router,
                        "--layers",
                        *layers,
                        "--output",
                        directory,
                        "--max-seq-len",
                        self.settings.context.max_seq_len,
                        "--op-package-host",
                        c.op_package_host,
                        "--op-package-device",
                        c.op_package_device,
                        *binding,
                    ),
                    (
                        directory / "GRAPHS.json",
                        directory / "context0.bin",
                        *(
                            directory / f"layer{layer}_{method}.io"
                            for layer in layers
                            for method in (
                                "prefill_attention",
                                "prefill_router",
                                "decode",
                            )
                        ),
                    ),
                    (directory / "methods.pte",),
                )
            stage(
                "runtime-assets",
                "export/runtime_assets.py",
                (
                    "--model",
                    c.model,
                    "--output",
                    out / "qnn-shared",
                    "--capacity",
                    self.settings.context.max_seq_len,
                ),
                tuple(
                    out / "qnn-shared" / name
                    for name in (
                        "cache-encoding.bin",
                        "cos.f32",
                        "sin.f32",
                        "RUNTIME.json",
                    )
                ),
            )
        if "vision" in c.components:
            stage(
                "vision",
                "export/vision.py",
                (
                    *common,
                    "--model",
                    c.model,
                    "--output",
                    out / "vision",
                    "--calib-images",
                    *c.images,
                    "--calibration-device",
                    c.calibration_device,
                    "--image-height",
                    c.image_height,
                    "--image-width",
                    c.image_width,
                    *binding,
                ),
                (
                    out / "vision/VISION.json",
                    out / "vision/vision.bin",
                    *((out / "vision/vision.pte",) if c.format == "pte" else ()),
                ),
                (out / "vision/vision.pte",) if c.format == "qnn" else (),
            )
        if text and c.format == "pte":
            for group in range(10):
                outputs = [
                    out / f"shared-model/group{group}" / name
                    for name in (
                        "graph.qnn",
                        "weights.bin",
                        "WEIGHTS.json",
                        "GRAPHS.json",
                    )
                ]
                outputs.extend(
                    out / f"shared-model/group{group}/layer{layer}_{method}.io"
                    for layer in range(group * 4, group * 4 + 4)
                    for method in ("prefill_attention", "prefill_router", "decode")
                )
                head = ("--head-context", out / "head/head.bin") if group == 9 else ()
                if head:
                    outputs.extend(
                        out / "shared-model/head" / name
                        for name in ("graph.qnn", "weights.bin", "WEIGHTS.json")
                    )
                    outputs.extend(
                        out / "shared-model" / name
                        for name in (
                            "cache-encoding.bin",
                            "cos.f32",
                            "sin.f32",
                            "RUNTIME.json",
                        )
                    )
                stage(
                    f"separate-group{group}",
                    "export/capture.py",
                    (
                        *common,
                        "--source-dir",
                        out / "qnn-shared",
                        "--output",
                        out / "shared-model",
                        "--groups",
                        group,
                        *head,
                        "--build-dir",
                        c.build_dir,
                        "--runtime-device-dir",
                        c.runtime_device_dir,
                        "--device",
                        c.device,
                        "--remote-root",
                        c.remote_root,
                    ),
                    outputs,
                )
            for phase in c.phases:
                lengths = (
                    range(128, self.settings.context.max_prefill_tokens + 1, 128)
                    if phase == "prefill"
                    and self.settings.schedule.prefill_order == "layer_major"
                    else (128,)
                )
                for length in lengths:
                    name = f"prefill-{length}" if phase == "prefill" else "decode"
                    path = out / "plans" / f"{name}.pte"
                    stage(
                        name,
                        "export/program.py",
                        (
                            *common,
                            "--quiet",
                            "--phase",
                            phase,
                            "--shared-model",
                            out / "shared-model",
                            "--output",
                            path,
                            "--prefill-tokens",
                            length,
                            "--cache-encoding",
                            out / "shared-model/cache-encoding.bin",
                        ),
                        (path, path.with_suffix(".json")),
                    )
        return stages

    def describe(self):
        return json.dumps(
            dict(
                configuration=self.config.to_dict(),
                stages=[
                    dict(
                        name=s.name,
                        command=list(s.command),
                        outputs=list(map(str, s.outputs)),
                        cleanup=list(map(str, s.cleanup)),
                    )
                    for s in self.stages
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )

    def checkpoint_shards(self):
        raw_text = (
            self.config.source == "safetensors" and "text" in self.config.components
        )
        if not raw_text and "vision" not in self.config.components:
            return []
        weights = json.loads(
            (self.config.model / "model.safetensors.index.json").read_text()
        )["weight_map"]
        names = {
            shard
            for key, shard in weights.items()
            if raw_text or key.startswith("model.visual.")
        }
        return [self.config.model / name for name in sorted(names)]

    def check(self):
        c = self.config
        required = [c.model / "config.json"]
        if c.qnn_sdk_root is None:
            raise ValueError("Set qnn_sdk_root or QNN_SDK_ROOT")
        required += [
            c.qnn_sdk_root / "include/QNN/QnnInterface.h",
            c.qnn_sdk_root / "lib/x86_64-linux-clang/libQnnSystem.so",
            c.qnn_sdk_root / "lib/x86_64-linux-clang/libQnnHtp.so",
        ]
        if c.qnn_binding:
            required.append(c.qnn_binding)
        if c.build and "text" in c.components:
            for name, directory in (
                ("hexagon_sdk_root", c.hexagon_sdk_root),
                ("hexagon_tools_root", c.hexagon_tools_root),
                ("android_ndk_root", c.android_ndk_root),
            ):
                if directory is None or not directory.is_dir():
                    raise ValueError(f"Set a valid {name} for the runtime build")
        elif "text" in c.components:
            required.append(c.op_package_host)
            if c.format == "pte":
                required += [
                    c.build_dir / "capture_weights",
                    c.build_dir
                    / f"op-package/hexagon-{c.dsp_arch}/libQnnDeltaNetOpPackage_HTP.so",
                ]
        if "vision" in c.components:
            required.extend(c.images)
            required.append(c.model / "model.safetensors.index.json")
        if "text" in c.components:
            if c.source == "safetensors":
                if c.corpus is None:
                    raise ValueError("Raw safetensors preparation requires corpus")
                required += [c.corpus, c.model / "model.safetensors.index.json"]
            else:
                required += [
                    c.calibration_dir / name
                    for name in (
                        "QUANTIZATION_COMPLETE.json",
                        "MODEL_CONFIG.json",
                        "cold-experts.i32",
                        "head/head-qdq.pt2",
                        "head/GPTQ.json",
                        (
                            "embedding.int4"
                            if self.settings.quantization.embedding == "int4_g32"
                            else "embedding.bf16"
                        ),
                    )
                ]
                for layer in range(40):
                    required += [
                        c.calibration_dir / f"layers/layer{layer}" / name
                        for name in (
                            "front-qdq.pt2",
                            "decode-qdq.pt2",
                            "moe-constants.npz",
                        )
                    ]
                    required += [
                        c.calibration_dir / f"banks/layer{layer}/BANK.json",
                        c.calibration_dir / f"banks/layer{layer}/experts.n2",
                        c.calibration_dir / f"indices/layer{layer}/bank-index.bin",
                    ]
                    if self.settings.quantization.router_shared == "w8a16_gptq":
                        required += [
                            c.calibration_dir / f"router/layer{layer}" / name
                            for name in (
                                "router-ar128-qdq.pt2",
                                "router-ar1-qdq.pt2",
                                "REPORT.json",
                            )
                        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing export inputs:\n" + "\n".join(missing))
        missing = [str(p) for p in self.checkpoint_shards() if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoint shards:\n" + "\n".join(missing))
        text_config = json.loads((c.model / "config.json").read_text())["text_config"]
        if tuple(
            text_config.get(key)
            for key in (
                "num_hidden_layers",
                "hidden_size",
                "num_experts",
                "num_experts_per_tok",
            )
        ) != (40, 2048, 256, 8):
            raise ValueError(
                "Export requires the 40-layer, hidden-2048, Top8/256 MoE checkpoint"
            )
        if "text" in c.components and c.source == "calibrated":
            prepared = json.loads((c.calibration_dir / "MODEL_CONFIG.json").read_text())
            if prepared["quantization"] != self.settings.to_dict()[
                "quantization"
            ] or any(
                prepared["cache"][key] != value
                for key, value in self.settings.to_dict()["cache"].items()
                if key.endswith("dtype")
            ):
                raise ValueError(
                    "Calibrated weight/cache recipes differ from model_config; prepare matching weights"
                )

    def environment(self):
        c = self.config
        env = dict(os.environ)
        for name, value in (
            ("QNN_SDK_ROOT", c.qnn_sdk_root),
            ("HEXAGON_SDK_ROOT", c.hexagon_sdk_root),
            ("HEXAGON_TOOLS_ROOT", c.hexagon_tools_root),
            ("ANDROID_NDK_ROOT", c.android_ndk_root),
            ("MOE_BUILD_DIR", c.build_dir),
            ("MOE_SOC_MODEL", c.soc_model),
            ("MOE_DSP_ARCH", c.dsp_arch.removeprefix("v")),
        ):
            if value is not None:
                env[name] = str(value)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(ROOT), str(REPO.parent), env.get("PYTHONPATH")))
        )
        libraries = []
        if c.android_ndk_root:
            libraries.append(
                c.android_ndk_root
                / "toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu"
            )
        if c.qnn_sdk_root:
            libraries.append(c.qnn_sdk_root / "lib/x86_64-linux-clang")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [*map(str, libraries), env.get("LD_LIBRARY_PATH", "")])
        )
        return env

    def identity(self):
        c = self.config
        inputs = [
            c.model / "config.json",
            c.model / "model.safetensors.index.json",
            *c.images,
        ]
        inputs += self.checkpoint_shards()
        if c.source == "safetensors":
            if c.corpus:
                inputs.append(c.corpus)
        elif "text" in c.components:
            inputs += [
                p
                for p in c.calibration_dir.rglob("*")
                if p.is_file()
                and p.suffix in (".pt2", ".npz")
                and p.relative_to(c.calibration_dir).parts[0]
                in ("layers", "router", "head")
            ]
            inputs += [
                c.calibration_dir / name
                for name in (
                    "MODEL_CONFIG.json",
                    "DATA.json",
                    "QUANTIZATION_COMPLETE.json",
                )
            ]
            inputs += [
                c.calibration_dir / name
                for name in ("embedding.int4", "embedding.bf16", "cold-experts.i32")
            ]
            for layer in range(40):
                inputs += [
                    c.calibration_dir / f"banks/layer{layer}" / name
                    for name in ("BANK.json", "experts.n2")
                ]
                inputs.append(
                    c.calibration_dir / f"indices/layer{layer}/bank-index.bin"
                )
        if c.qnn_binding:
            inputs.append(c.qnn_binding)
        if c.qnn_sdk_root:
            inputs.append(c.qnn_sdk_root / "include/QNN/QnnInterface.h")
            inputs += [
                c.qnn_sdk_root / "lib/x86_64-linux-clang" / name
                for name in ("libQnnHtp.so", "libQnnSystem.so")
            ]
        if not c.build and "text" in c.components:
            inputs += [
                c.op_package_host,
                c.build_dir / "capture_weights",
                c.build_dir
                / f"op-package/hexagon-{c.dsp_arch}/libQnnDeltaNetOpPackage_HTP.so",
            ]
        source_roots = (
            ROOT,
            REPO
            / "examples/qualcomm/custom_op/deltanet_op_package_htp/DeltaNetOpPackage",
            REPO / "examples/qualcomm/oss_scripts/llama/model",
        )
        sources = sorted(
            p
            for root in source_roots
            for p in root.rglob("*")
            if (
                p.suffix in (".py", ".cpp", ".h", ".idl", ".sh", ".mk", ".xml")
                or p.name in ("Makefile", "CMakeLists.txt")
            )
            and "tests" not in p.parts
        )
        code_hash = hashlib.sha256(
            b"".join(
                str(p.relative_to(REPO)).encode() + p.read_bytes() for p in sources
            )
        ).hexdigest()
        return dict(
            configuration=c.to_dict(),
            code_sha256=code_hash,
            inputs={str(p): file_stamp(p) for p in inputs if p.is_file()},
        )

    def finish(self):
        c = self.config
        out = c.output
        if "text" in c.components:
            for name in (
                "banks",
                "indices",
                "cold-experts.i32",
                "embedding.int4",
                "embedding.bf16",
                "embedding.json",
            ):
                source = c.calibration_dir / name
                target = out / name
                if not source.exists():
                    continue
                if target.is_symlink() and target.resolve() == source.resolve():
                    continue
                if target.exists() or target.is_symlink():
                    raise ValueError(
                        f"Refusing to replace an unrelated artifact: {target}"
                    )
                target.symlink_to(source, target_is_directory=source.is_dir())
            (out / "assets").mkdir(exist_ok=True)
            for name in ("cache-encoding.bin", "cos.f32", "sin.f32"):
                shutil.copy2(out / "qnn-shared" / name, out / "assets" / name)
        if "prefill" in c.phases and "text" in c.components and c.format == "pte":
            lengths = (
                list(range(128, self.settings.context.max_prefill_tokens + 1, 128))
                if self.settings.schedule.prefill_order == "layer_major"
                else [128]
            )
            write_json(
                out / "plans/PLANS.json",
                dict(
                    version=1,
                    ar_len=128,
                    capacity=self.settings.context.max_seq_len,
                    max_prefill_tokens=self.settings.context.max_prefill_tokens,
                    prefill_order=self.settings.schedule.prefill_order,
                    lengths=lengths,
                    model_configuration=self.settings.to_dict(),
                ),
            )
        inference_config = None
        if (
            "text" in c.components
            and c.format == "pte"
            and set(c.phases) == {"prefill", "decode"}
        ):
            inference_config = out / "device.json"
            profile = dict(
                model_config=str(out / "MODEL_CONFIG.json"),
                soc_model=c.soc_model,
                dsp_arch=c.dsp_arch,
                device=c.device,
                model=str(c.model),
                artifacts=str(out),
                build_dir=str(c.build_dir),
                shared_model=str(out / "shared-model"),
                embedding_table=(
                    str(out / "embedding.int4")
                    if self.settings.quantization.embedding == "int4_g32"
                    else None
                ),
                vision_artifacts=str(out / "vision"),
                local_decode_pte=str(out / "plans/decode.pte"),
                prefill_pte=c.remote_root + "/model/prefill/prefill.pte",
                decode_pte=c.remote_root + "/model/decode/decode.pte",
                runtime_device_dir=c.runtime_device_dir,
                remote_weights=c.remote_root + "/weights",
                remote_root=c.remote_root + "/runtime",
            )
            write_json(inference_config, profile)
        return dict(
            status="complete",
            format=c.format,
            components=list(c.components),
            phases=list(c.phases),
            model_configuration=self.settings.to_dict(),
            source=str(c.calibration_dir),
            weight_mode="external_shared" if c.format == "pte" else "shared_context",
            shared_contexts=(
                str(out / ("shared-model" if c.format == "pte" else "qnn-shared"))
                if "text" in c.components
                else None
            ),
            inference_config=str(inference_config) if inference_config else None,
            external_weight_links="banks, indices and embedding reference calibration_dir; retain that directory",
        )

    def run(self, resume=True):
        self.check()
        out = self.config.output
        out.mkdir(parents=True, exist_ok=True)
        with (out / ".export.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "An export is already running in this directory"
                ) from error
            identity = self.identity()
            record = out / "EXPORT_STATE.json"
            if not record.exists() and any(
                p.name != ".export.lock" for p in out.iterdir()
            ):
                raise ValueError(
                    "Output is not an empty export workspace; use a new directory"
                )
            if (
                not record.exists()
                and self.config.source == "safetensors"
                and "text" in self.config.components
                and self.config.calibration_dir.exists()
                and any(self.config.calibration_dir.iterdir())
            ):
                raise ValueError(
                    "Raw preparation requires a new calibration_dir; use source=calibrated to reuse existing weights"
                )
            state = (
                json.loads(record.read_text())
                if record.exists()
                else dict(identity=identity, stages={})
            )
            if state["identity"] != identity:
                raise ValueError(
                    "Export inputs/configuration/code changed; use a new output directory"
                )
            write_json(out / "MODEL_CONFIG.json", self.settings.to_dict())
            write_json(out / "EXPORT_PLAN.json", json.loads(self.describe()))
            dirty = not resume
            for stage in self.stages:
                previous = state["stages"].get(stage.name, {})
                valid = previous.get("status") == "complete" and all(
                    p.is_file() for p in stage.outputs
                )
                if valid:
                    valid = previous["outputs"] == {
                        str(p): file_stamp(p) for p in stage.outputs
                    }
                if not dirty and valid:
                    print("EXPORT_REUSED", stage.name, flush=True)
                    continue
                dirty = True
                (out / "EXPORT.json").unlink(missing_ok=True)
                log = out / "logs" / f"{stage.name}.log"
                log.parent.mkdir(exist_ok=True)
                started = time.monotonic()
                state["stages"][stage.name] = dict(status="running", log=str(log))
                write_json(record, state)
                print("EXPORT_START", stage.name, "log=" + str(log), flush=True)
                try:
                    with log.open("w") as stream:
                        result = subprocess.run(
                            stage.command,
                            cwd=ROOT,
                            env=self.environment(),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),),
                        )
                    if result.returncode:
                        raise RuntimeError(
                            f"{stage.name} exited with {result.returncode}; see {log}"
                        )
                    if any(not p.is_file() for p in stage.outputs):
                        raise RuntimeError(
                            f"{stage.name} did not produce its declared outputs; see {log}"
                        )
                    for path in stage.cleanup:
                        path.unlink(missing_ok=True)
                    state["stages"][stage.name] = dict(
                        status="complete",
                        seconds=time.monotonic() - started,
                        outputs={str(p): file_stamp(p) for p in stage.outputs},
                        log=str(log),
                    )
                    write_json(record, state)
                except BaseException as error:
                    state["stages"][stage.name].update(
                        status="failed", error=str(error)
                    )
                    write_json(record, state)
                    raise
                print("EXPORT_DONE", stage.name, flush=True)
            write_json(out / "EXPORT.json", self.finish())
        print("EXPORT_COMPLETE", out, flush=True)
        return out / "EXPORT.json"
