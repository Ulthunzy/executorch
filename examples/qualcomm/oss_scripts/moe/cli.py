"""Common command-line entry points for registered MoE models."""

import argparse
import json
from pathlib import Path

from executorch.examples.qualcomm.oss_scripts.moe.inference.base import MoeRuntime
from executorch.examples.qualcomm.oss_scripts.moe import create_model
from executorch.examples.qualcomm.oss_scripts.moe.target import SOC_MODELS


def select_model(path, requested, default):
    profile = json.loads(path.read_text()) if path else {}
    if not isinstance(profile, dict):
        raise ValueError("Expected a JSON object for the model configuration")
    return create_model(requested or profile.get("model_family", default))


def export_main(argv=None, *, default_model="qwen35"):
    selection = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selection.add_argument("--model-family")
    selection.add_argument("--config", type=Path)
    family, _ = selection.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        description="Export a registered Qualcomm MoE model"
    )
    parser.add_argument("--model-family", default=family.model_family)
    try:
        model = select_model(family.config, family.model_family, default_model)
        parser.add_argument("--config", type=Path, help="Optional export profile")
        parser.add_argument("-a", "--artifact", "--output", dest="output", type=Path)
        parser.add_argument(
            "-s", "--device", help="ADB serial for shared-weight capture"
        )
        parser.add_argument("-m", "--soc-model", "--soc_model", choices=SOC_MODELS)
        parser.add_argument("--model", "--model_path", type=Path)
        parser.add_argument(
            "--source", choices=("safetensors", "calibrated"), help=argparse.SUPPRESS
        )
        parser.add_argument("--calibration-dir", type=Path)
        parser.add_argument("--build-dir", type=Path, help="Native MoE build directory")
        parser.add_argument(
            "-b",
            "--build_folder",
            dest="executorch_android_build",
            type=Path,
            help="Prebuilt ExecuTorch Android runtime directory",
        )
        parser.add_argument("--qnn-sdk-root", "--qnn_sdk_root", type=Path)
        parser.add_argument("--corpus", type=Path)
        vision_data = parser.add_mutually_exclusive_group()
        vision_data.add_argument(
            "--image_path",
            dest="images",
            type=Path,
            nargs="+",
            help="Vision calibration images",
        )
        vision_data.add_argument(
            "--calib-image-dataset",
            "--calib_image_dataset",
            choices=("coco128", "coco8"),
            help="Vision calibration dataset (default: coco128); downloaded and cached automatically",
        )
        parser.add_argument("--max_seq_len", "--max-seq-len", type=int)
        parser.add_argument("--prefill_ar_len", "--prefill-ar-len", type=int)
        parser.add_argument("--cache-experts", "--cache_experts", type=int)
        parser.add_argument(
            "--quantization-layer-batch",
            type=int,
            choices=(1, 2, 3, 4),
            help="Concurrent weight-quantization layers (default: 4)",
        )
        parser.add_argument(
            "--calibration-dtype",
            choices=("bfloat16", "float32"),
            help="Original-model calibration precision (default: bfloat16)",
        )
        parser.add_argument("--components", nargs="+", choices=("text", "vision"))
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--print-plan", action="store_true")
        mode.add_argument("--check", action="store_true")
        parser.add_argument("--no-resume", action="store_true")
        args = parser.parse_args(argv)
        config = model.load_export_config(
            args.config,
            args.output,
            device=args.device,
            soc_model=args.soc_model,
            components=args.components,
            model=args.model,
            source=args.source,
            calibration_dir=args.calibration_dir,
            build_dir=args.build_dir,
            executorch_android_build=args.executorch_android_build,
            qnn_sdk_root=args.qnn_sdk_root,
            corpus=args.corpus,
            images=args.images,
            max_seq_len=args.max_seq_len,
            prefill_ar_len=args.prefill_ar_len,
            cache_experts=args.cache_experts,
            calib_image_dataset=args.calib_image_dataset,
            quantization_layer_batch=args.quantization_layer_batch,
            calibration_dtype=args.calibration_dtype,
        )
        workflow = model.create_export_workflow(config)
        if args.print_plan:
            print(workflow.describe())
        elif args.check:
            workflow.check()
            print("EXPORT_PREREQUISITES_OK")
        else:
            return workflow.run(resume=not args.no_resume)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Export failed: {error}\n")


def run_main(argv=None, *, default_model="qwen35"):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-family")
    parser.add_argument("--config", type=Path)
    selection, remaining = parser.parse_known_args(argv)
    try:
        model = select_model(selection.config, selection.model_family, default_model)
        if selection.config:
            remaining = ["--config", str(selection.config), *remaining]
        args = model.parse_runtime_args(remaining)
        return MoeRuntime(model.create_inference_backend(args)).run()
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Inference failed: {error}\n")
