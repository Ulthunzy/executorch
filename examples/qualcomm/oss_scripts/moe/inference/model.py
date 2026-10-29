"""Qwen3.5 graph, image and deployment contracts."""

from executorch.examples.qualcomm.oss_scripts.moe.inference.base import (
    InferenceBackend,
    MoeRuntime,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.artifacts import (
    check_artifacts,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.diagnostic import (
    run_diagnostic,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.plans import (
    load_prefill_bundle,
    select_plans,
)
from executorch.examples.qualcomm.oss_scripts.moe.inference.decoder import run_session


class ModelInference(InferenceBackend):
    def validate_artifacts(self):
        check_artifacts(self.args)

    def prepare(self):
        args = self.args
        metadata = None
        if args.with_vision:
            from executorch.examples.qualcomm.oss_scripts.moe.inference.encoder import (
                vision_metadata,
            )

            metadata = vision_metadata(args.model, args.vision_artifacts)
            args.model_settings.target.check_manifest(metadata)
            if args.model_settings.quantization.vision.endswith("_gptq") and (
                not metadata.get("gptq")
                or not all(
                    x.get("method") == "fixed_grid_gptq" for x in metadata["gptq"]
                )
                or not any(
                    x.get("module") == "visual.pos_embed" for x in metadata["gptq"]
                )
            ):
                raise ValueError(
                    "Selected vision recipe requires calibrated GPTQ weights"
                )
        if args.image_path and not args.image_path.is_file():
            raise FileNotFoundError(args.image_path)
        return metadata if args.image_path else None

    def check_plans(self):
        select_plans(self.args)
        load_prefill_bundle(self.args)

    def run_chat(self):
        return run_session(self.args)

    def run_diagnostic(self, metadata):
        return run_diagnostic(self.args, metadata)


def inference(args):
    return MoeRuntime(ModelInference(args)).run(validate=False)
