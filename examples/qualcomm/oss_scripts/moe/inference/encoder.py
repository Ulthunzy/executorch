import hashlib
import json
from pathlib import Path


def vision_metadata(model: Path, artifacts: Path):
    metadata = json.loads((artifacts / "VISION.json").read_text())
    config_bytes = (model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    vision = config["vision_config"]
    height, width = metadata["image_height"], metadata["image_width"]
    patch, merge = vision["patch_size"], vision["spatial_merge_size"]
    patches = height // patch * (width // patch)
    if (
        metadata["version"] != 1
        or metadata.get("precision") != "W8A16"
        or metadata["model_config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
        or metadata["image_token_id"] != config["image_token_id"]
        or config["image_token_id"] != 248056
        or min(height, width) < patch * merge
        or height % (patch * merge)
        or width % (patch * merge)
        or metadata["input_shape"]
        != [
            1,
            patches,
            vision["in_channels"] * vision["temporal_patch_size"] * patch**2,
        ]
        or metadata["output_shape"] != [1, patches // merge**2, 2048]
    ):
        raise ValueError("Vision artifacts do not match the model and image dimensions")
    if not (artifacts / "vision.bin").is_file():
        raise FileNotFoundError(artifacts / "vision.bin")
    return metadata


def pad_prompt(tokenizer, ids, target_tokens, ar_len=128, max_tokens=None):
    target = target_tokens or ((len(ids) + ar_len - 1) // ar_len * ar_len)
    if (
        target < len(ids)
        or (max_tokens is not None and target > max_tokens)
        or target % ar_len
    ):
        raise ValueError(
            f"Prompt needs {len(ids)} tokens; choose an AR{ar_len} multiple through {max_tokens}"
        )
    if target == len(ids):
        return ids
    space = tokenizer.encode(" ", add_special_tokens=False)
    if len(space) != 1:
        raise ValueError("Expected a single whitespace token")
    # Pad the system message so the final state remains at the generation prompt.
    end = ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return ids[:end] + space * (target - len(ids)) + ids[end:]


def image_prompt(
    tokenizer,
    prompt,
    image_tokens,
    target_tokens,
    enable_thinking=False,
    system_prompt="你是一个有帮助的助手，请根据图片和用户的问题准确回答。",
    max_tokens=None,
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt.replace("<image>", "").strip()},
            ],
        },
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if ids.count(image_id) != 1:
        raise ValueError("Expected one image placeholder in the chat template")
    index = ids.index(image_id)
    ids[index : index + 1] = [image_id] * image_tokens
    unfilled = len(ids)
    return (
        messages,
        pad_prompt(tokenizer, ids, target_tokens, max_tokens=max_tokens),
        unfilled,
    )


def prepare_image(args, ids, metadata, run_dir, np):
    from PIL import Image
    from executorch.examples.qualcomm.oss_scripts.llama.dataset.preprocessors import (
        _qwen_vl_image_to_patches,
    )

    config = json.loads((args.model / "config.json").read_text())
    vision = config["vision_config"]
    height, width = metadata["image_height"], metadata["image_width"]
    with Image.open(args.image_path) as image:
        pixels = _qwen_vl_image_to_patches(image, height, width).unsqueeze(0)
    np.asarray(pixels, dtype="<f4").tofile(run_dir / "pixels.f32")
    grid_h = height // (vision["patch_size"] * vision["spatial_merge_size"])
    grid_w = width // (vision["patch_size"] * vision["spatial_merge_size"])
    image_id = config["image_token_id"]
    if ids.count(image_id) != grid_h * grid_w:
        raise ValueError("Image token count differs from the vision output")
    positions, cursor, position = [], 0, 0
    while cursor < len(ids):
        if ids[cursor] != image_id:
            positions.append([position] * 3)
            cursor += 1
            position += 1
            continue
        if ids[cursor : cursor + grid_h * grid_w] != [image_id] * (grid_h * grid_w):
            raise ValueError("Image placeholders must be contiguous")
        positions.extend(
            [position, position + h, position + w]
            for h in range(grid_h)
            for w in range(grid_w)
        )
        cursor += grid_h * grid_w
        position += max(grid_h, grid_w)
    delta = position - len(ids)
    from executorch.examples.qualcomm.oss_scripts.moe.utils import runtime_assets

    base_cos = np.fromfile(runtime_assets(args) / "cos.f32", dtype="<f4").reshape(
        -1, 32
    )
    base_sin = np.fromfile(runtime_assets(args) / "sin.f32", dtype="<f4").reshape(
        -1, 32
    )
    if base_cos.shape != base_sin.shape or len(base_cos) < max(
        2048, args.decode_capacity
    ):
        raise ValueError("Insufficient base RoPE tables")
    positions.extend([i + delta] * 3 for i in range(len(ids), len(base_cos)))
    positions = np.asarray(positions, dtype=np.int64).T
    sections = config["text_config"]["rope_parameters"]["mrope_section"]
    axes = np.zeros(32, dtype=np.int64)
    axes[1 : min(sections[1] * 3, 32) : 3] = 1
    axes[2 : min(sections[2] * 3, 32) : 3] = 2
    for name, base in [("cos", base_cos), ("sin", base_sin)]:
        columns = np.arange(32)
        mixed = base[positions[axes].T, columns]
        np.asarray(mixed, dtype="<f4").tofile(run_dir / f"{name}.f32")
    np.asarray(positions, dtype="<i4").tofile(run_dir / "mrope-positions.i32")
    return dict(
        image_path=str(args.image_path.resolve()),
        image_tokens=grid_h * grid_w,
        grid_thw=[1, height // vision["patch_size"], width // vision["patch_size"]],
        mrope_delta=delta,
        encoder=metadata,
    )
