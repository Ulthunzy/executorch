import json


def runtime_assets(args):
    if args.shared_model:
        manifest = args.shared_model / "group0/GRAPHS.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get(
            "prefill_valid_length"
        ):
            return args.shared_model
    return args.artifacts / "assets"


def validate_rope(directory, capacity):
    cos = directory / "cos.f32"
    sin = directory / "sin.f32"
    if (
        not cos.is_file()
        or not sin.is_file()
        or cos.stat().st_size != sin.stat().st_size
        or cos.stat().st_size % (32 * 4)
        or cos.stat().st_size < capacity * 32 * 4
    ):
        raise ValueError(f"{directory}: RoPE tables do not cover {capacity} positions")
