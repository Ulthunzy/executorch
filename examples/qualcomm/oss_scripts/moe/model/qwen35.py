from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import warnings
from pathlib import Path

from executorch.examples.qualcomm.oss_scripts.moe.target import TargetConfig

DEFAULT_MODEL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/qwen35_35b_a3b/model.json"
)


from executorch.examples.qualcomm.oss_scripts.moe.quantize.strategy import (
    QuantizationConfig,
)


@dataclass(frozen=True)
class CacheConfig:
    kv_dtype: str = "uint16"
    gdn_state_dtype: str = "uint16"
    conv_state_dtype: str = "uint16"
    append_only: bool = True
    state_swap: bool = True


@dataclass(frozen=True)
class ContextConfig:
    max_seq_len: int = 1152
    prefill_ar_len: int = 128


@dataclass(frozen=True)
class GenerationConfig:
    enable_thinking: bool = True
    system_prompt: str = "你是一个有帮助的人工智能助手。"
    max_new_tokens: int = 128
    temperature: float = 0
    ignore_eos: bool = False
    interactive: bool = False
    stream: bool = True


@dataclass(frozen=True)
class ScheduleConfig:
    prefill_order: str = "layer_major"
    cache_experts: int = 32
    prefill_async_io: bool = True
    startup_prefetch: str = "overlap"
    io_workers: int = 8
    startup_io_workers: int = 4
    read_chunk_kib: int = 256
    decode_hit_first: bool = True
    decode_ready_io: bool = True
    overlap_feedback: bool = True
    context_load_mode: str = "auto"


@dataclass(frozen=True)
class ModelConfig:
    target: TargetConfig = field(default_factory=TargetConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    def validate(self):
        self.target.validate()
        q, c, x, g, s = (
            self.quantization,
            self.cache,
            self.context,
            self.generation,
            self.schedule,
        )
        integers = (
            q.expert_bits,
            q.expert_activation_bits,
            q.expert_gu_group,
            q.expert_down_group,
            q.reconstruction_steps,
            q.reconstruction_min_steps,
            q.reconstruction_patience,
            x.max_seq_len,
            x.prefill_ar_len,
            g.max_new_tokens,
            s.cache_experts,
            s.io_workers,
            s.startup_io_workers,
            s.read_chunk_kib,
        )
        if any(type(v) is not int for v in integers):
            raise ValueError(
                "Bit widths, groups, lengths, counts and reconstruction steps must be integers"
            )
        if any(
            type(v) not in (int, float)
            for v in (
                q.expert_gptq_damping,
                q.router_gptq_damping,
                q.matrix_gptq_damping,
                q.reconstruction_min_delta,
                *(
                    ()
                    if q.reconstruction_target_nmse is None
                    else (q.reconstruction_target_nmse,)
                ),
            )
        ):
            raise ValueError(
                "GPTQ damping and reconstruction tolerances must be numeric"
            )
        if q.attention not in {"source_qdq", "gptq"}:
            raise ValueError(
                "Attention is loaded from calibrated QDQ; requantize the source to change its precision"
            )
        if q.vision not in {"w8a16", "w8a16_gptq"}:
            raise ValueError("The vision exporter currently supports the W8A16 recipe")
        if q.head not in {"source_qdq", "w4a16_gptq"}:
            raise ValueError("Unsupported head recipe")
        if q.calibration_dtype not in {"bfloat16", "float32"}:
            raise ValueError("calibration_dtype must be bfloat16 or float32")
        if (
            q.expert_bits,
            q.expert_activation_bits,
            q.expert_gu_group,
            q.expert_down_group,
        ) != (2, 8, 1024, 512):
            raise ValueError(
                "Native MoE requires reconstructed N2/A8, GU group 1024 and down group 512"
            )
        if q.router_shared not in {"w8a16_gptq", "fp16"} or q.embedding not in {
            "int4_g32",
            "bf16",
        }:
            raise ValueError("Unsupported Router/shared or embedding recipe")
        if (
            any(
                not math.isfinite(v) or v <= 0
                for v in (
                    q.expert_gptq_damping,
                    q.router_gptq_damping,
                    q.matrix_gptq_damping,
                )
            )
            or q.reconstruction_steps < 0
            or q.reconstruction_min_steps < 0
            or q.reconstruction_patience < 0
            or not math.isfinite(q.reconstruction_min_delta)
            or not 0 <= q.reconstruction_min_delta < 1
            or (
                q.reconstruction_target_nmse is not None
                and (
                    not math.isfinite(q.reconstruction_target_nmse)
                    or q.reconstruction_target_nmse < 0
                )
            )
        ):
            raise ValueError(
                "Use positive GPTQ damping, nonnegative reconstruction limits and a finite relative min_delta in [0, 1)"
            )
        if (
            c.kv_dtype != "uint16"
            or c.conv_state_dtype != "uint16"
            or c.gdn_state_dtype not in {"uint16", "float16"}
        ):
            raise ValueError(
                "KV/conv caches require UINT16; GDN state supports UINT16 or typed FP16"
            )
        if (
            not 1024 <= x.max_seq_len <= 32768
            or x.max_seq_len % 128
            or x.prefill_ar_len != 128
        ):
            raise ValueError(
                "Use AR128 and an aligned context from 1024 through 32768 tokens"
            )
        if (
            not isinstance(g.system_prompt, str)
            or not g.system_prompt
            or not 1 <= g.max_new_tokens <= x.max_seq_len
        ):
            raise ValueError(
                "Generation needs a nonempty system prompt and valid output length"
            )
        if (
            type(g.temperature) not in (int, float)
            or not math.isfinite(g.temperature)
            or g.temperature < 0
        ):
            raise ValueError("temperature must be finite and nonnegative")
        if s.prefill_order not in {
            "layer_major",
            "chunk_major",
        } or s.cache_experts not in {32, 64}:
            raise ValueError(
                "Use layer_major prefill (chunk_major is a deprecated alias) "
                "and 32/64 expert slots per layer"
            )
        if (
            s.io_workers not in {4, 8, 16}
            or not 1 <= s.startup_io_workers <= 16
            or s.read_chunk_kib not in {64, 128, 256, 512, 1024}
        ):
            raise ValueError("Invalid expert IO workers or read chunk size")
        if s.context_load_mode not in {
            "auto",
            "async",
            "serial",
        } or s.startup_prefetch not in {"none", "sync", "overlap"}:
            raise ValueError("Invalid context/startup loading policy")
        if not s.prefill_async_io and s.startup_prefetch != "none":
            raise ValueError(
                "Startup prefetch requires prefill_async_io; select startup_prefetch=none"
            )
        for value in (
            q.expert_gptq_only,
            c.append_only,
            c.state_swap,
            g.enable_thinking,
            g.ignore_eos,
            g.interactive,
            g.stream,
            s.prefill_async_io,
            s.decode_hit_first,
            s.decode_ready_io,
            s.overlap_feedback,
        ):
            if type(value) is not bool:
                raise ValueError("Boolean model settings must be JSON true/false")
        if s.decode_ready_io and not s.decode_hit_first:
            raise ValueError("decode_ready_io requires decode_hit_first")
        if s.prefill_order == "chunk_major":
            warnings.warn(
                "chunk_major is deprecated and will be removed in a future release; "
                "using the unified layer_major prefill scheduler",
                FutureWarning,
                stacklevel=2,
            )
            return replace(self, schedule=replace(s, prefill_order="layer_major"))
        return self

    def to_dict(self):
        data = dict(version=1, **asdict(self))
        data["target"] = self.target.to_dict()
        return data

    def compile_settings(self):
        return dict(
            quantization={
                "attention": self.quantization.attention,
                "router_shared": self.quantization.router_shared,
            },
            cache={k: v for k, v in asdict(self.cache).items() if k.endswith("dtype")},
            context=asdict(self.context),
        )

    @property
    def prefill_workspace_tokens(self):
        return 1024

    def matches_compile_settings(self, compiled):
        if not isinstance(compiled, dict) or not isinstance(
            compiled.get("context"), dict
        ):
            return False
        context = dict(compiled["context"])
        # Older manifests included a host workspace limit in the graph identity.
        context.pop("max_prefill_tokens", None)
        return {**compiled, "context": context} == self.compile_settings()


def load_model_config(path=DEFAULT_MODEL_CONFIG):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or type(data.get("version", 1)) is not int
        or data.get("version", 1) != 1
    ):
        raise ValueError("Expected a version 1 model configuration object")
    unknown = set(data) - {
        "version",
        "target",
        "quantization",
        "cache",
        "context",
        "generation",
        "schedule",
    }
    if unknown:
        raise ValueError(f"Unknown model configuration sections: {sorted(unknown)}")
    context = dict(data.get("context", {}))
    context.pop("max_prefill_tokens", None)
    try:
        return ModelConfig(
            target=TargetConfig.resolve(**data.get("target", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
            cache=CacheConfig(**data.get("cache", {})),
            context=ContextConfig(**context),
            generation=GenerationConfig(**data.get("generation", {})),
            schedule=ScheduleConfig(**data.get("schedule", {})),
        ).validate()
    except TypeError as error:
        raise ValueError(f"Invalid model configuration: {error}") from error


def add_model_config_argument(parser):
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)


def shared_model_signature(root):
    signature = {}
    for group in range(10):
        directory = root / f"group{group}"
        metadata = json.loads((directory / "GRAPHS.json").read_text())
        signature[f"group{group}"] = dict(
            contexts=metadata["contexts"],
            io={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(directory.glob("*.io"))
            },
        )
    signature["head"] = json.loads((root / "head/WEIGHTS.json").read_text())[
        "source_sha256"
    ]
    return signature


from executorch.examples.qualcomm.oss_scripts.moe import register_model
from executorch.examples.qualcomm.oss_scripts.moe.model.base import (
    ModelAdapter,
    ModelArchitecture,
)


class Qwen35Adapter(ModelAdapter):
    architecture = ModelArchitecture(40, 4, 2048, 256, 8)

    @property
    def root(self):
        return Path(__file__).resolve().parents[1]

    @property
    def default_export_config(self):
        from executorch.examples.qualcomm.oss_scripts.moe.wrappers.config import (
            DEFAULT_EXPORT_CONFIG,
        )

        return DEFAULT_EXPORT_CONFIG

    def load_export_config(self, path, output=None, **overrides):
        from executorch.examples.qualcomm.oss_scripts.moe.wrappers.config import (
            load_export_config,
        )

        return load_export_config(path, output, **overrides)

    def create_export_workflow(self, config):
        from executorch.examples.qualcomm.oss_scripts.moe.wrappers.model_wrapper import (
            ModelWrapper,
        )

        return ModelWrapper(config, self)

    def parse_runtime_args(self, argv):
        from executorch.examples.qualcomm.oss_scripts.moe.inference.config import (
            parse_args,
        )

        return parse_args(argv)

    def create_inference_backend(self, args):
        from executorch.examples.qualcomm.oss_scripts.moe.inference.model import (
            ModelInference,
        )

        return ModelInference(args)


register_model("qwen35", Qwen35Adapter)
