import json

from workflow.artifacts import check_artifacts
from workflow.config import parse_args
from workflow.inference import inference


def main(argv=None):
    args = parse_args(argv)
    if args.print_config:
        print(json.dumps(args.model_settings.to_dict(), ensure_ascii=False, indent=2))
        return
    check_artifacts(args)
    return inference(args)


def run_inference(
    *,
    config=None,
    weights_path=None,
    device=None,
    soc_model=None,
    prompt="请介绍一下自己。",
    temperature=None,
    ignore_eos=None,
    image_path=None,
    with_vision=None,
    interactive=None,
    stream=None,
    enable_thinking=None,
    max_new_tokens=None,
):
    """Run a native session; omitted controls inherit the selected configuration."""
    argv = ["--prompt", prompt]
    for flag, value in (
        ("--config", config),
        ("--weights-path", weights_path),
        ("--device", device),
        ("--soc-model", soc_model),
        ("--image-path", image_path),
        ("--temperature", temperature),
        ("--max-new-tokens", max_new_tokens),
    ):
        if value is not None:
            argv += [flag, str(value)]
    for name, value in (
        ("ignore-eos", ignore_eos),
        ("interactive", interactive),
        ("stream", stream),
        ("with-vision", with_vision),
        ("enable-thinking", enable_thinking),
    ):
        if value is not None:
            if type(value) is not bool:
                raise ValueError(f"{name} must be bool or None")
            argv.append("--" + ("" if value else "no-") + name)
    return main(argv)


if __name__ == "__main__":
    main()
