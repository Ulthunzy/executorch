"""Deploy once, then keep the native tokenizer/model session alive across turns."""

from datetime import datetime
import json
import shlex

from .device import AndroidDevice, DevicePaths
from .deployment import deploy_artifacts
from .plans import select_plans


def session_command(args, paths, image=None):
    command = [
        paths.prefill + "/qwen35_moe_runner",
        "--model-root",
        paths.root,
        "--tokenizer-path",
        args.remote_root + "/tokenizer.json",
        "--shared-model",
        paths.shared_model,
        "--model-config",
        paths.root + "/SESSION_CONFIG.json",
        "--prompt",
        args.prompt,
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--system-prompt",
        args.system_prompt,
        "--log-path",
        paths.root + "/chat.log",
        "--stream" if args.stream else "--no-stream",
        "--vision" if args.with_vision else "--no-vision",
        "--enable-thinking" if args.enable_thinking else "--no-enable-thinking",
        "--ignore-eos" if args.ignore_eos else "--no-ignore-eos",
    ]
    if args.prompt_tokens is not None:
        command += ["--prompt-tokens", str(args.prompt_tokens)]
    command.append("--interactive" if args.interactive else "--no-interactive")
    if image is not None:
        command += ["--image-path", image]
    return "cd " + shlex.quote(paths.prefill) + " && " + shlex.join(command)


def run_session(args):
    device = AndroidDevice(args.device)
    device.check_target(args.model_settings.target)
    label = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-chat"
    directory = args.output / label
    directory.mkdir(parents=True)
    paths = DevicePaths(
        args.remote_root + "/" + label, args.remote_root + "/shared-model"
    )
    record = dict(
        mode="chat",
        model_configuration=args.model_settings.to_dict(),
        serial=args.device,
        weights=str(args.artifacts),
        image=str(args.image_path) if args.image_path else None,
        vision_enabled=args.with_vision,
        exit_code=None,
    )
    print("RESULTS", directory, flush=True)
    try:
        select_plans(args)
        deploy_artifacts(args, device, paths, directory, None)
        settings = directory / "SESSION_CONFIG.json"
        settings.write_text(
            json.dumps(args.model_settings.to_dict(), ensure_ascii=False, indent=2)
            + "\n"
        )
        device.adb("push", settings, paths.root + "/SESSION_CONFIG.json")
        image = None
        if args.image_path:
            image = paths.root + "/input-image" + args.image_path.suffix.lower()
            device.adb("push", args.image_path, image)
        command = session_command(args, paths, image)
        record["command"] = command
        record["exit_code"] = device.terminal(command, directory / "console.log")
        if record["exit_code"]:
            raise RuntimeError(
                f"Device chat exited with {record['exit_code']}; see {directory}"
            )
        return directory
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = str(error)
        raise
    finally:
        (directory / "RUN.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
        try:
            device.adb("pull", paths.root + "/chat.log", directory / "runner.log")
        except Exception as error:
            print(f"Could not collect native log: {error}", flush=True)
