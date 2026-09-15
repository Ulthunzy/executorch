"""Local Hugging Face causal language models without implicit config overrides."""

import json
from pathlib import Path

import torch


class HFEngine:
    """Load a local model strictly and expose fixed-input next-token logits."""

    def __init__(
        self, model_path, *, device="cuda:0", dtype="float32", config_override=None
    ):
        from transformers import AutoConfig, AutoModelForCausalLM

        directory = Path(model_path)
        if not directory.is_dir():
            raise ValueError("HF --model must be a local from_pretrained directory")
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("HF supports float32 or bfloat16")
        overrides = config_override or {}
        if "model_type" in overrides:
            values = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            values.update(overrides)
            model_type = values.pop("model_type")
            config = AutoConfig.for_model(model_type, **values)
        else:
            config = AutoConfig.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False
            )
            config.update(overrides)
        self.device = torch.device(device)
        self.model, loading = AutoModelForCausalLM.from_pretrained(
            directory,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32 if dtype == "float32" else torch.bfloat16,
            output_loading_info=True,
        )
        if any(
            loading.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incomplete HF model loading: {loading}")
        self.model.to(self.device).eval()
        loading = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in loading.items()
        }
        self.details = dict(
            config=self.model.config.to_dict(),
            loading=loading,
            config_override=overrides,
        )
        self.vocab_size = self.model.get_input_embeddings().weight.shape[0]
        self.max_input_length = self.details["config"].get(
            "max_position_embeddings", self.details["config"].get("n_positions")
        )

    @torch.inference_mode()
    def __call__(self, ids):
        """Evaluate the fixed prefix with no generation processors or KV reuse."""
        inputs = torch.as_tensor(ids, dtype=torch.int64, device=self.device)[None]
        if inputs.numel() < 1 or inputs.min() < 0 or inputs.max() >= self.vocab_size:
            raise ValueError("HF input token ID outside embedding vocabulary")
        if (
            self.max_input_length is not None
            and inputs.shape[1] > self.max_input_length
        ):
            raise ValueError("HF input exceeds model position capacity")
        return self.model(input_ids=inputs, use_cache=False, return_dict=True).logits[0]
