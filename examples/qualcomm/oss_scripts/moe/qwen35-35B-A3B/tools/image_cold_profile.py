#!/usr/bin/env python3
"""Materialize the SM8850 image startup-cache sidecar, optionally rebuilding its independent-image ranking."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]
sys.path.insert(0, str(ROOT))

from workflow.image_cold_profile import (  # noqa: E402
    DEFAULT_PROFILE,
    SOURCE_IDENTITIES,
    STRATEGIES,
    materialize_image_cold_profile,
)


def rebuild_profile(collection):
    collection = Path(collection).resolve()
    record = json.loads(collection.read_text())
    if len(record["runs"]) != 3:
        raise ValueError("Exactly three independent image runs are required")
    profile = json.loads(DEFAULT_PROFILE.read_text())
    by_source = {row["source"]["sha256"]: row for row in record["runs"]}
    if set(by_source) != {sha for _, sha in SOURCE_IDENTITIES}:
        raise ValueError("Collection must contain only the three approved images")
    images = []
    for index, (relative, expected_sha) in enumerate(SOURCE_IDENTITIES):
        image = REPO / relative
        if hashlib.sha256(image.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("Calibration source image changed: " + relative)
        row = by_source[expected_sha]
        if (
            row["source"]["repository_path"] != relative
            or row["label"] != f"image{index}"
        ):
            raise ValueError("Calibration source identity/label mismatch")
        routes = (
            collection.parent / "collection" / row["label"] / "prefill-all-routes.i32"
        )
        data = routes.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if (
            routes.is_symlink()
            or digest != row["route_sha256"]
            or len(data) != 81920 * 4
        ):
            raise ValueError("Independent calibration route shape/checksum mismatch")
        values = struct.unpack("<81920i", data)
        counts = [[0] * 256 for _ in range(40)]
        for chunk in range(2):
            for layer in range(40):
                begin = (chunk * 40 + layer) * 128 * 8
                for token in range(128):
                    ids = values[begin + token * 8 : begin + (token + 1) * 8]
                    if len(set(ids)) != 8 or any(
                        expert < 0 or expert >= 256 for expert in ids
                    ):
                        raise ValueError("Invalid independent calibration Top8 routes")
                    for expert in ids:
                        counts[layer][expert] += 1
        images.append(counts)
        profile["sources"][index]["routes_sha256"] = digest

    def rank(training, strategy):
        result = []
        for layer in range(40):
            counts = [
                sum(image[layer][expert] for image in training) for expert in range(256)
            ]
            presence = [
                sum(image[layer][expert] > 0 for image in training)
                for expert in range(256)
            ]
            if strategy == "token_frequency":
                order = sorted(range(256), key=lambda expert: (-counts[expert], expert))
            else:
                order = sorted(
                    range(256),
                    key=lambda expert: (-presence[expert], -counts[expert], expert),
                )
            result.append(order[:64])
        return result

    scores = {}
    for strategy in STRATEGIES:
        folds = []
        prefix32 = 0
        for heldout in range(3):
            order = rank(
                [image for index, image in enumerate(images) if index != heldout],
                strategy,
            )
            folds.append(
                sum(
                    images[heldout][layer][expert] > 0
                    for layer in range(40)
                    for expert in order[layer]
                )
            )
            prefix32 += sum(
                images[heldout][layer][expert] > 0
                for layer in range(40)
                for expert in order[layer][:32]
            )
        scores[strategy] = {
            "fold_unique_hits": folds,
            "total_unique_hits": sum(folds),
            "prefix32_total_unique_hits": prefix32,
        }
    selected = max(
        STRATEGIES, key=lambda strategy: scores[strategy]["total_unique_hits"]
    )
    profile["loo"] = scores
    profile["selected_strategy"] = selected
    profile["experts"] = rank(images, selected)
    data = struct.pack(
        "<2560i", *(expert for row in profile["experts"] for expert in row)
    )
    profile["table_sha256"] = hashlib.sha256(data).hexdigest()
    profile.pop("calibration_evidence_sha256", None)
    profile["calibration_collection_sha256"] = hashlib.sha256(
        collection.read_bytes()
    ).hexdigest()
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--collection",
        type=Path,
        help="Rebuild only from the three independent calibration route files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the two optional image sidecars",
    )
    args = parser.parse_args(argv)
    if args.collection:
        if args.profile != DEFAULT_PROFILE:
            parser.error("--collection uses the predefined independent-image profile")
        profile = rebuild_profile(args.collection)
        with tempfile.TemporaryDirectory(prefix="qwen-image-cold-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile, ensure_ascii=False) + "\n")
            assets = materialize_image_cold_profile(
                args.output_dir, path, overwrite=args.overwrite
            )
    else:
        assets = materialize_image_cold_profile(
            args.output_dir, args.profile, overwrite=args.overwrite
        )
    print(
        json.dumps(
            {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in assets
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
