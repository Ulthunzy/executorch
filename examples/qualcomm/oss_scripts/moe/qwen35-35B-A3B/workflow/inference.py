"""One inference function for device chat and diagnostic execution."""

from .diagnostic import run_diagnostic
from .plans import load_prefill_bundle, select_plans
from .session import run_session


def inference(args):
    metadata = None
    if args.with_vision:
        from workflow.vision import vision_metadata

        metadata = vision_metadata(args.model, args.vision_artifacts)
        args.model_settings.target.check_manifest(metadata)
        if args.model_settings.quantization.vision.endswith("_gptq") and (
            not metadata.get("gptq")
            or not all(x.get("method") == "fixed_grid_gptq" for x in metadata["gptq"])
            or not any(x.get("module") == "visual.pos_embed" for x in metadata["gptq"])
        ):
            raise ValueError("Selected vision recipe requires calibrated GPTQ weights")
    if args.image_path and not args.image_path.is_file():
        raise FileNotFoundError(args.image_path)
    if args.check:
        select_plans(args)
        load_prefill_bundle(args)
        print(f"Local artifacts and optimized runners ready: {args.artifacts}")
        return
    if args.runner == "diagnostic":
        return run_diagnostic(args, metadata if args.image_path else None)
    return run_session(args)
