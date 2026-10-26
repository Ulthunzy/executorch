"""Export the configured Qwen MoE components with one serial workflow."""

import argparse
from pathlib import Path

from workflow.export_config import DEFAULT_EXPORT_CONFIG, load_export_config
from workflow.export_pipeline import ExportPipeline
from workflow.target import DSP_ARCHES, SOC_MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPORT_CONFIG)
    parser.add_argument("--output", type=Path, help="Override the output directory")
    parser.add_argument("--device", help="ADB serial used for shared-weight capture")
    parser.add_argument("--soc-model", "--soc_model", choices=SOC_MODELS)
    parser.add_argument("--dsp-arch", choices=DSP_ARCHES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--check", action="store_true", help="Check local prerequisites")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        pipeline = ExportPipeline(
            load_export_config(
                args.config,
                args.output,
                device=args.device,
                soc_model=args.soc_model,
                dsp_arch=args.dsp_arch,
            )
        )
        if args.print_plan:
            print(pipeline.describe())
        elif args.check:
            pipeline.check()
            print("EXPORT_PREREQUISITES_OK")
        else:
            pipeline.run(resume=not args.no_resume)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
