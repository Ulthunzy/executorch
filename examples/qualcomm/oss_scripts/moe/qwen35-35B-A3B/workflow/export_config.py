"""Export job settings; numerical model settings remain in model_config.py."""

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re

from workflow.model_config import ModelConfig, load_model_config
from workflow.target import TargetConfig

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]
DEFAULT_EXPORT_CONFIG = ROOT / "configs/export_sm8975.json"


@dataclass(frozen=True)
class ExportConfig:
    soc_model: str
    model_settings: ModelConfig
    model_config: Path
    model: Path
    calibration_dir: Path
    output: Path
    source: str
    components: tuple[str, ...]
    phases: tuple[str, ...]
    format: str
    build: bool
    deploy_runtime: bool
    prefill_gdn_core: str
    gdn_state_vtcm: bool
    build_dir: Path
    qnn_binding: Path | None
    op_package_host: Path
    op_package_device: str
    device: str | None
    runtime_device_dir: str | None
    remote_root: str
    qnn_sdk_root: Path | None
    hexagon_sdk_root: Path | None
    hexagon_tools_root: Path | None
    android_ndk_root: Path | None
    corpus: Path | None
    images: tuple[Path, ...]
    calibration_device: str
    expert_batch: int
    image_height: int
    image_width: int
    vision_source: Path | None = None

    @property
    def dsp_arch(self):
        return self.model_settings.target.dsp_arch

    def to_dict(self):
        data = asdict(self)
        data["dsp_arch"] = self.dsp_arch
        data["model_settings"] = self.model_settings.to_dict()
        return json.loads(json.dumps(data, default=str))


def load_export_config(
    path=DEFAULT_EXPORT_CONFIG,
    output=None,
    *,
    device=None,
    soc_model=None,
    components=None,
):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    allowed = (set(ExportConfig.__dataclass_fields__) - {"model_settings"}) | {
        "dsp_arch"
    }
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        raise ValueError("Expected a version 1 export configuration")
    unknown = set(data) - allowed - {"version"}
    if unknown:
        raise ValueError(f"Unknown export settings: {sorted(unknown)}")

    def location(name, default=None, base=REPO):
        value = data.get(name, default)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty path")
        target = Path(value)
        return (target if target.is_absolute() else base / target).resolve()

    config_path = location("model_config", "model.json", path.parent)
    if config_path is None:
        raise ValueError("model_config must name the numerical recipe")
    settings = load_model_config(config_path)
    selected_soc = soc_model or data.get("soc_model", settings.target.soc_model)
    selected_arch = data.get("dsp_arch") if soc_model is None else None
    settings = replace(
        settings, target=TargetConfig.resolve(selected_soc, selected_arch)
    ).validate()
    required = ("model", "calibration_dir", "output", "build_dir")
    if any(not data.get(key) for key in required):
        raise ValueError(f"Export requires paths: {', '.join(required)}")
    components = (
        components
        if components is not None
        else data.get("components", ["text", "vision"])
    )
    phases = data.get("phases", ["prefill", "decode"])
    for name, values, supported in (
        ("components", components, {"text", "vision"}),
        ("phases", phases, {"prefill", "decode"}),
    ):
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or value not in supported for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"{name} must select distinct values from {sorted(supported)}"
            )
    source = data.get("source", "calibrated")
    format_name = data.get("format", "pte")
    if source not in ("calibrated", "safetensors") or format_name not in ("pte", "qnn"):
        raise ValueError("Use source=calibrated/safetensors and format=pte/qnn")
    if format_name == "qnn" and set(phases) != {"prefill", "decode"}:
        raise ValueError(
            "Shared QNN contexts contain both phases; phases selects PTE plans only"
        )
    build = data.get("build", True)
    if type(build) is not bool:
        raise ValueError("build must be a JSON boolean")
    deploy_runtime = data.get("deploy_runtime", False)
    if type(deploy_runtime) is not bool:
        raise ValueError("deploy_runtime must be a JSON boolean")
    prefill_gdn_core = data.get("prefill_gdn_core", "chunked_mixed_fp16")
    if prefill_gdn_core not in (
        "chunked_mixed_fp16",
        "custom_fp16",
        "custom_fp32",
        "custom_u16",
    ):
        raise ValueError("Unsupported prefill_gdn_core")
    gdn_state_vtcm = data.get("gdn_state_vtcm", False)
    if type(gdn_state_vtcm) is not bool:
        raise ValueError("gdn_state_vtcm must be a JSON boolean")
    if gdn_state_vtcm and prefill_gdn_core in ("custom_fp16", "custom_fp32"):
        raise ValueError("GDN VTCM layout requires a UINT16 core")
    if settings.cache.gdn_state_dtype == "float16" and (
        source != "calibrated"
        or prefill_gdn_core != "chunked_mixed_fp16"
        or gdn_state_vtcm
    ):
        raise ValueError(
            "Typed FP16 state requires calibrated weights, chunked prefill and DDR state"
        )
    calibration_device = data.get("calibration_device", "cuda")
    if calibration_device not in ("cuda", "cpu"):
        raise ValueError("calibration_device must be cuda or cpu")
    if (
        source == "safetensors"
        and "text" in components
        and (
            calibration_device != "cuda"
            or settings.quantization.attention != "gptq"
            or not settings.quantization.expert_gptq_only
            or settings.quantization.embedding != "int4_g32"
            or settings.quantization.router_shared != "w8a16_gptq"
        )
    ):
        raise ValueError(
            "Raw text preparation requires CUDA, all-GPTQ weights and INT4 embedding"
        )
    if "text" in components and settings.schedule.context_load_mode == "serial":
        raise ValueError("Shared text contexts require auto/async context loading")
    if "text" in components and settings.quantization.head != "w4a16_gptq":
        raise ValueError("The shared text exporter requires a GPTQ W4A16 head")
    dimensions = {name: data.get(name, 448) for name in ("image_height", "image_width")}
    if any(
        type(value) is not int or value < 32 or value % 32
        for value in dimensions.values()
    ):
        raise ValueError("Image dimensions must be positive multiples of 32")
    batch = data.get("expert_batch", 8)
    if type(batch) is not int or batch not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Invalid expert_batch")
    images = data.get("images", [])
    if not isinstance(images, list) or any(
        not isinstance(p, str) or not p for p in images
    ):
        raise ValueError("images must contain calibration image paths")
    vision_source = location("vision_source")
    if "vision" in components and not images and vision_source is None:
        raise ValueError("Vision export requires calibration images")
    build_dir = location("build_dir")
    device = device if device is not None else data.get("device")
    if device is not None and (not isinstance(device, str) or not device):
        raise ValueError("device must be an ADB serial string")
    runtime = data.get("runtime_device_dir")
    remote = data.get("remote_root", "/data/local/tmp/qwen35-moe-export")
    if not isinstance(remote, str) or not remote:
        raise ValueError("remote_root must be an absolute device path")
    if data.get("op_package_device", "default") is None:
        raise ValueError("op_package_device must name the deployed package")
    for name, value in (
        ("remote_root", remote),
        ("runtime_device_dir", runtime),
        (
            "op_package_device",
            data.get(
                "op_package_device",
                "/data/local/tmp/qwen35-35B-A3B/libQnnDeltaNetOpPackage_HTP.so",
            ),
        ),
    ):
        if value is not None and (
            not isinstance(value, str)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
            or ".." in Path(value).parts
        ):
            raise ValueError(f"Invalid absolute device path: {name}")
    if not remote.startswith("/data/local/tmp/"):
        raise ValueError("Use a dedicated remote_root under /data/local/tmp")
    if (
        "text" in components
        and format_name == "pte"
        and (
            not isinstance(device, str)
            or not device
            or not isinstance(runtime, str)
            or not runtime
        )
    ):
        raise ValueError(
            "External shared PTE export requires device and runtime_device_dir for SDK weight capture"
        )
    if deploy_runtime and (not device or not runtime):
        raise ValueError("Runtime deployment requires device and runtime_device_dir")
    if deploy_runtime and not runtime.startswith("/data/local/tmp/"):
        raise ValueError("Use a dedicated runtime_device_dir under /data/local/tmp")
    config = ExportConfig(
        soc_model=settings.target.soc_model,
        model_settings=settings,
        model_config=config_path,
        model=location("model"),
        calibration_dir=location("calibration_dir"),
        output=Path(output).resolve() if output is not None else location("output"),
        source=source,
        components=tuple(components),
        phases=tuple(phases),
        format=format_name,
        build=build,
        deploy_runtime=deploy_runtime,
        prefill_gdn_core=prefill_gdn_core,
        gdn_state_vtcm=gdn_state_vtcm,
        build_dir=build_dir,
        qnn_binding=location("qnn_binding"),
        op_package_host=location(
            "op_package_host",
            str(build_dir / "op-package/x86_64-linux-clang/libQnnDeltaNetOpPackage.so"),
        ),
        op_package_device=data.get(
            "op_package_device",
            "/data/local/tmp/qwen35-35B-A3B/libQnnDeltaNetOpPackage_HTP.so",
        ),
        device=device,
        runtime_device_dir=runtime,
        remote_root=remote,
        qnn_sdk_root=location("qnn_sdk_root", os.environ.get("QNN_SDK_ROOT")),
        hexagon_sdk_root=location(
            "hexagon_sdk_root", os.environ.get("HEXAGON_SDK_ROOT")
        ),
        hexagon_tools_root=location(
            "hexagon_tools_root", os.environ.get("HEXAGON_TOOLS_ROOT")
        ),
        android_ndk_root=location(
            "android_ndk_root", os.environ.get("ANDROID_NDK_ROOT")
        ),
        corpus=location("corpus"),
        images=tuple((REPO / p).resolve() for p in images),
        calibration_device=calibration_device,
        expert_batch=batch,
        image_height=dimensions["image_height"],
        image_width=dimensions["image_width"],
        vision_source=vision_source,
    )
    if config.output == config.calibration_dir:
        raise ValueError(
            "Use a separate output directory to preserve calibration artifacts"
        )
    if config.output.is_relative_to(
        config.calibration_dir
    ) or config.calibration_dir.is_relative_to(config.output):
        raise ValueError("Use non-overlapping export and calibration directories")
    return config
