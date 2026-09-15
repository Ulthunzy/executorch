"""Explicit tool registry; adding a stage does not require changing the scheduler."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]


@dataclass(frozen=True)
class Tool:
    script: Path
    description: str


TOOLS = {
    "hf": Tool(
        ROOT / "benchmarks/hf_benchmark.py",
        "SafeTensors CUDA 对话基线（Qwen3/Lincal，温度 0）",
    ),
    "pte": Tool(ROOT / "benchmarks/benchmark.py", "PTE 手机对话测试、复核汇总与比较"),
    "pt2": Tool(ROOT / "benchmarks/pt2_benchmark.py", "固定形状 QDQ PT2 CUDA 对话测试"),
    "sample": Tool(
        ROOT / "benchmarks/sample_benchmark.py", "外部 benchmark sample bundle 测试"
    ),
    "ppl": Tool(ROOT / "ppl/evaluate.py", "冻结 PPL 语料及 HF/PT2/PTE 标准 PPL"),
    "check-inputs": Tool(
        ROOT / "common/check_inputs.py", "冻结 ChatML/token IDs，并可核对手机 tokenizer"
    ),
    "calibration-build": Tool(
        ROOT / "calibration/build_think_calibration.py", "校准候选选择、审核快照、合并"
    ),
    "calibration-check": Tool(
        ROOT / "calibration/validate_think_calibration.py",
        "校准数据加载与 think/EOS 完整性检查",
    ),
    "build-driver": Tool(
        ROOT / "benchmarks/build_driver.py", "构建隔离的手机生成/PPL 测试驱动"
    ),
    "report": Tool(
        ROOT / "reporting/compare.py", "跨模型汇总，保留缺失结果和待复核状态"
    ),
    "errors": Tool(
        ROOT / "reporting/errors.py", "按会话整理输入、实际回答、参考答案及待复核项"
    ),
    "export": Tool(
        ROOT.parent / "llama.py", "原有 Qualcomm 量化/PTE 导出入口（显式选择才执行）"
    ),
}
