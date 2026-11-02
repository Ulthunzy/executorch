from dataclasses import dataclass


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
    reconstruction_min_steps: int = 20
    reconstruction_patience: int = 10
    reconstruction_min_delta: float = 1e-4
    reconstruction_target_nmse: float | None = None
    expert_gptq_only: bool = False
    router_shared: str = "w8a16_gptq"
    router_gptq_damping: float = 0.01
    embedding: str = "int4_g32"
    calibration_dtype: str = "bfloat16"
