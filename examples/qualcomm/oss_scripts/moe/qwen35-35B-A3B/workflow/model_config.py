from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

from workflow.target import TargetConfig

DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parents[1] / "configs/model.json"


@dataclass(frozen=True)
class QuantizationConfig:
    attention: str = "source_qdq"
    vision: str = "w8a16"
    head: str = "source_qdq"
    matrix_gptq_damping: float = 0.01
    expert_bits: int = 2
    expert_activation_bits: int = 8
    expert_gu_group: int = 1024
    expert_down_group: int = 512
    expert_gptq_damping: float = 0.1
    reconstruction_steps: int = 80
    expert_gptq_only: bool = False
    router_shared: str = "w8a16_gptq"
    router_gptq_damping: float = 0.01
    embedding: str = "int4_g32"


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
    max_prefill_tokens: int = 1024


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
            x.max_seq_len,
            x.prefill_ar_len,
            x.max_prefill_tokens,
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
            )
        ):
            raise ValueError("GPTQ damping must be numeric")
        if q.attention not in {"source_qdq", "gptq"}:
            raise ValueError(
                "Attention is loaded from calibrated QDQ; requantize the source to change its precision"
            )
        if q.vision not in {"w8a16", "w8a16_gptq"}:
            raise ValueError("The vision exporter currently supports the W8A16 recipe")
        if q.head not in {"source_qdq", "w4a16_gptq"}:
            raise ValueError("Unsupported head recipe")
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
        ):
            raise ValueError(
                "GPTQ damping must be positive and reconstruction_steps nonnegative"
            )
        if (c.kv_dtype, c.gdn_state_dtype, c.conv_state_dtype) != ("uint16",) * 3:
            raise ValueError(
                "This MoE runner has UINT16 KV/GDN/conv interfaces; UINT8 requires new calibration and runner support"
            )
        if (
            not 1024 <= x.max_seq_len <= 32768
            or x.max_seq_len % 128
            or x.prefill_ar_len != 128
            or not 128 <= x.max_prefill_tokens < x.max_seq_len
            or x.max_prefill_tokens % 128
            or (s.prefill_order == "layer_major" and x.max_prefill_tokens > 1024)
        ):
            raise ValueError(
                "Use AR128, aligned context up to 32768 and a smaller aligned prefill limit; "
                "layer_major Native batches support at most 1024 tokens"
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
                "Use layer_major/chunk_major prefill and 32/64 expert slots per layer"
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
    try:
        return ModelConfig(
            target=TargetConfig.resolve(**data.get("target", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
            cache=CacheConfig(**data.get("cache", {})),
            context=ContextConfig(**data.get("context", {})),
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
