"""Qwen3.5 graph plans and artifact contracts for the shared export workflow."""

import hashlib
import json
import os
import shutil
import sys

from executorch.examples.qualcomm.oss_scripts.moe.wrappers.base_component import (
    ExportWorkflow,
    Stage,
)
from executorch.examples.qualcomm.oss_scripts.moe.wrappers.config import (
    REPO,
    ROOT,
    ExportConfig,
)
from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import write_json
from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import (
    input_stamp as file_stamp,
)
from executorch.examples.qualcomm.oss_scripts.moe.quantize.cache import (
    prepared_artifacts,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.sdk_runtime import (
    runtime_files,
)
from executorch.examples.qualcomm.oss_scripts.moe.model.qwen35 import (
    load_model_config,
)


class ModelWrapper(ExportWorkflow):
    def __init__(self, config: ExportConfig, model=None):
        from executorch.examples.qualcomm.oss_scripts.moe import create_model

        self.model = model or create_model()
        self.architecture = self.model.architecture
        super().__init__(config, ROOT)

    def plan(self):
        c = self.config
        out = c.output
        snapshot = out / "MODEL_CONFIG.json"
        common = ("--model-config", str(snapshot))
        binding = ("--qnn-binding", str(c.qnn_binding)) if c.qnn_binding else ()
        stages = []

        def stage(
            name,
            script,
            arguments,
            outputs,
            cleanup=(),
            *,
            requires=None,
            publishes=(),
            resource="cpu",
            needs_device=False,
        ):
            command = (sys.executable, "-u", str(ROOT / script), *map(str, arguments))
            stages.append(
                Stage(
                    name,
                    command,
                    tuple(outputs),
                    tuple(cleanup),
                    always_run=name in ("runtime", "vision-data"),
                    preserve_outputs=name == "quantize",
                    requires=requires,
                    publishes=publishes,
                    resource=resource,
                    needs_device=needs_device,
                )
            )

        if "vision" in c.components and c.calib_image_dataset and not c.vision_source:
            stage(
                "vision-data",
                "quantize/calibration_images.py",
                ("--dataset", c.calib_image_dataset, "--cache", c.calib_dataset_dir),
                (c.calib_dataset_dir / c.calib_image_dataset / "DATASET.json",),
            )

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
                            "libet_infer.so",
                            "libet_qnnop.so",
                            "capture_weights",
                            "TARGET.json",
                            "NATIVE_TARGET.json",
                            "op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so",
                        )
                    ),
                )
            )
        text = "text" in c.components
        pipeline = text and c.source == "safetensors"
        if pipeline:
            stage(
                "quantize",
                "quantize/prepare_model.py",
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
                    "--layer-batch",
                    c.quantization_layer_batch,
                    "--device",
                    c.calibration_device,
                    *binding,
                ),
                prepared_artifacts(c.calibration_dir, self.architecture.layers),
                publishes=tuple(
                    f"layer{layer}-router" for layer in range(self.architecture.layers)
                )
                + ("head",),
                resource="gpu" if c.calibration_device == "cuda" else "cpu",
            )
        if text:
            stage(
                "head",
                "wrappers/head.py",
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
                requires=("quantize:head",) if pipeline else None,
            )
            for group in range(self.architecture.groups):
                directory = out / f"qnn-shared/group{group}"
                layers = range(
                    group * self.architecture.group_size,
                    min(
                        (group + 1) * self.architecture.group_size,
                        self.architecture.layers,
                    ),
                )
                router = (
                    ("--router-qdq", c.calibration_dir / "router")
                    if self.settings.quantization.router_shared == "w8a16_gptq"
                    else ()
                )
                stage(
                    f"qnn-group{group}",
                    "wrappers/shared.py",
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
                        "--prefill-gdn-core",
                        c.prefill_gdn_core,
                        "--attention-core",
                        c.attention_core,
                        *(("--gdn-state-vtcm",) if c.gdn_state_vtcm else ()),
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
                    requires=(
                        tuple(f"quantize:layer{layer}-router" for layer in layers)
                        if pipeline
                        else None
                    ),
                )
            stage(
                "runtime-assets",
                "wrappers/runtime_assets.py",
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
                requires=(
                    (
                        "quantize",
                        "head",
                        *(
                            f"qnn-group{group}"
                            for group in range(self.architecture.groups)
                        ),
                    )
                    if pipeline
                    else None
                ),
            )
        if "vision" in c.components:
            vision_arguments = (
                "--model-config",
                snapshot,
                "--model",
                c.model,
                "--output",
                out / "vision",
                "--image-height",
                c.image_height,
                "--image-width",
                c.image_width,
            )
            stage(
                "vision",
                (
                    "inference/reuse_vision.py"
                    if c.vision_source
                    else "wrappers/vision.py"
                ),
                (
                    (*vision_arguments, "--source", c.vision_source)
                    if c.vision_source
                    else (
                        *common,
                        "--model",
                        c.model,
                        "--output",
                        out / "vision",
                        *(
                            (
                                "--calib-manifest",
                                c.calib_dataset_dir
                                / c.calib_image_dataset
                                / "DATASET.json",
                            )
                            if c.calib_image_dataset
                            else ("--calib-images", *c.images)
                        ),
                        "--calibration-device",
                        c.calibration_device,
                        "--image-height",
                        c.image_height,
                        "--image-width",
                        c.image_width,
                        *binding,
                    )
                ),
                (
                    out / "vision/VISION.json",
                    out / "vision/vision.bin",
                    *((out / "vision/vision.pte",) if c.format == "pte" else ()),
                    *((out / "vision/VISION_REUSE.json",) if c.vision_source else ()),
                ),
                (out / "vision/vision.pte",) if c.format == "qnn" else (),
            )
        if text and c.format == "pte":
            if c.deploy_runtime:
                stage(
                    "runtime",
                    ROOT / "inference/sdk_runtime.py",
                    (
                        "--sdk",
                        c.qnn_sdk_root,
                        "--soc-model",
                        c.soc_model,
                        "--device",
                        c.device,
                        "--remote",
                        c.runtime_device_dir,
                        "--manifest",
                        c.build_dir / "QNN_RUNTIME.json",
                    ),
                    (c.build_dir / "QNN_RUNTIME.json",),
                    needs_device=True,
                )
            for group in range(self.architecture.groups):
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
                    for layer in range(
                        group * self.architecture.group_size,
                        min(
                            (group + 1) * self.architecture.group_size,
                            self.architecture.layers,
                        ),
                    )
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
                    "wrappers/capture.py",
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
                    needs_device=True,
                )
            for phase in c.phases:
                workspace_tokens = (
                    self.settings.prefill_workspace_tokens
                    if phase == "prefill"
                    else 128
                )
                path = out / "plans" / f"{phase}.pte"
                stage(
                    phase,
                    "wrappers/program.py",
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
                        workspace_tokens,
                        "--cache-encoding",
                        out / "shared-model/cache-encoding.bin",
                    ),
                    (path, path.with_suffix(".json")),
                )
        return stages

    def checkpoint_shards(self):
        raw_text = (
            self.config.source == "safetensors" and "text" in self.config.components
        )
        raw_vision = (
            "vision" in self.config.components and self.config.vision_source is None
        )
        if not raw_text and not raw_vision:
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
        if c.deploy_runtime:
            required.extend(runtime_files(c.qnn_sdk_root, self.settings.target))
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
                    c.build_dir / "libet_qnnop.so",
                ]
        if "vision" in c.components and c.vision_source is None:
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
                        "DATA.json",
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
                for layer in range(self.architecture.layers):
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
        if "vision" in c.components and c.vision_source:
            from executorch.examples.qualcomm.oss_scripts.moe.inference.reuse_vision import (
                validate_vision_source,
            )

            validate_vision_source(
                c.vision_source, c.model, self.settings, c.image_height, c.image_width
            )
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
            from executorch.examples.qualcomm.oss_scripts.moe.quantize.provenance import (
                validate_checkpoint_source,
            )

            validate_checkpoint_source(
                c.model, json.loads((c.calibration_dir / "DATA.json").read_text())
            )
            prepared = load_model_config(
                c.calibration_dir / "MODEL_CONFIG.json"
            ).to_dict()
            if prepared["quantization"] != self.settings.to_dict()[
                "quantization"
            ] or any(
                prepared["cache"][key] != value
                for key, value in self.settings.to_dict()["cache"].items()
                if key.endswith("dtype")
                and not (
                    key == "gdn_state_dtype"
                    and value == "float16"
                    and prepared["cache"][key] == "uint16"
                )
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
            ("EXECUTORCH_ANDROID_BUILD", c.executorch_android_build),
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
            c.model / "tokenizer.json",
            c.model / "generation_config.json",
            *c.images,
        ]
        inputs += self.checkpoint_shards()
        if "vision" in c.components and c.vision_source:
            inputs += [
                c.vision_source / name
                for name in ("VISION.json", "vision.bin", "vision.pte")
            ]
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
            if c.deploy_runtime:
                inputs.extend(runtime_files(c.qnn_sdk_root, self.settings.target))
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
            for layer in range(self.architecture.layers):
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
                c.build_dir / "libet_qnnop.so",
            ]
        source_roots = (
            ROOT,
            ROOT / "model",
            ROOT / "runner",
            REPO / "examples/qualcomm/custom_op",
            REPO / "examples/qualcomm/oss_scripts/llama/model",
        )
        sources = sorted(
            p
            for root in source_roots
            for p in root.rglob("*")
            if (
                p.suffix
                in (
                    ".py",
                    ".cpp",
                    ".h",
                    ".idl",
                    ".sh",
                    ".mk",
                    ".xml",
                    ".exports",
                    ".cmake",
                )
                or p.name in ("Makefile", "CMakeLists.txt")
            )
            and "tests" not in p.parts
        )
        sources.append(REPO / "backends/qualcomm/quantizer/metrics.py")
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
            write_json(
                out / "plans/PLANS.json",
                dict(
                    version=3,
                    ar_len=128,
                    capacity=self.settings.context.max_seq_len,
                    workspace_tokens=self.settings.prefill_workspace_tokens,
                    prefill_order=self.settings.schedule.prefill_order,
                    prefill_pte="prefill.pte",
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
                model_family="qwen35",
                model_config=str(out / "MODEL_CONFIG.json"),
                soc_model=c.soc_model,
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

    def protected_paths(self):
        c = self.config
        return [
            p
            for p in (
                REPO,
                c.model,
                c.model_config,
                c.calibration_dir,
                *c.images,
                c.corpus,
                c.qnn_binding,
                c.qnn_sdk_root,
                c.hexagon_sdk_root,
                c.hexagon_tools_root,
                c.android_ndk_root,
                c.vision_source,
            )
            if p
        ]

    def preserved_paths(self):
        out = self.config.output
        return tuple(
            {
                out / path.relative_to(out).parts[0]
                for path in (self.config.calibration_dir, self.config.build_dir)
                if path != out and path.is_relative_to(out)
            }
        )

    def package_path(self):
        return self.config.output / "model.pte" if self.config.format == "pte" else None

    def package_identity(self):
        c = self.config
        profiles = dict(image_cold_profile_sha256=None, text_cold_profile_sha256=None)
        if c.soc_model == "SM8850" and "vision" in c.components:
            from executorch.examples.qualcomm.oss_scripts.moe.inference.image_cold_profile import (
                DEFAULT_PROFILE,
            )

            profiles["image_cold_profile_sha256"] = hashlib.sha256(
                DEFAULT_PROFILE.read_bytes()
            ).hexdigest()
        if c.soc_model == "SM8850" and "text" in c.components:
            from executorch.examples.qualcomm.oss_scripts.moe.inference.text_cold_profile import (
                DEFAULT_PROFILE,
            )

            profiles["text_cold_profile_sha256"] = hashlib.sha256(
                DEFAULT_PROFILE.read_bytes()
            ).hexdigest()
        return profiles

    def package(self):
        from executorch.examples.qualcomm.oss_scripts.moe.package import pack_export

        return pack_export(self.config)


ExportPipeline = ModelWrapper
