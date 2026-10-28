#!/usr/bin/env python3
"""Materialize the SM8850 text startup-cache sidecar or rebuild its independent-text ranking."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow.text_cold_profile import (  # noqa: E402
    DEFAULT_PROFILE,
    SOURCE_IDENTITIES,
    STRATEGIES,
    materialize_text_cold_profile,
)


def rebuild_profile(collection):
    collection = Path(collection).resolve()
    record = json.loads(collection.read_text())
    if len(record["runs"]) != 3:
        raise ValueError("Exactly three independent natural-text runs are required")
    profile = json.loads(DEFAULT_PROFILE.read_text())
    by_source = {row["source"]["sha256"]: row for row in record["runs"]}
    if set(by_source) != {sha for _, sha, _ in SOURCE_IDENTITIES}:
        raise ValueError(
            "Collection must contain only the three approved natural texts"
        )
    samples = []
    for index, (filename, expected_sha, prompt_sha) in enumerate(SOURCE_IDENTITIES):
        row = by_source[expected_sha]
        source = row["source"]
        if (
            source["filename"] != filename
            or row["label"] != f"text{index}"
            or source["prompt_utf8_sha256"] != prompt_sha
            or source["prompt"] + "\n" != profile["sources"][index]["content"]
            or (
                source["valid_tokens"],
                source["real_space_tokens_added"],
                source["inactive_padding_tokens"],
            )
            != (256, 0, 0)
        ):
            raise ValueError(
                "Natural calibration source identity/token policy mismatch"
            )
        directory = collection.parent / "collection" / row["label"]
        routes = directory / "prefill-all-routes.i32"
        tokens = directory / "tokens.i64"
        data = routes.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if (
            routes.is_symlink()
            or tokens.is_symlink()
            or digest != row["route_sha256"]
            or len(data) != 81920 * 4
            or len(tokens.read_bytes()) != 256 * 8
            or hashlib.sha256(tokens.read_bytes()).hexdigest()
            != profile["sources"][index]["expected_input_sha256"]
        ):
            raise ValueError(
                "Independent calibration route/input shape/checksum mismatch"
            )
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
        samples.append(counts)
        profile["sources"][index]["routes_sha256"] = digest

    def rank(training, strategy):
        result = []
        for layer in range(40):
            counts = [
                sum(sample[layer][expert] for sample in training)
                for expert in range(256)
            ]
            presence = [
                sum(sample[layer][expert] > 0 for sample in training)
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
                [sample for index, sample in enumerate(samples) if index != heldout],
                strategy,
            )
            folds.append(
                sum(
                    samples[heldout][layer][expert] > 0
                    for layer in range(40)
                    for expert in order[layer]
                )
            )
            prefix32 += sum(
                samples[heldout][layer][expert] > 0
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
    profile["experts"] = rank(samples, selected)
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
        help="Rebuild only from the three independent natural calibration texts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the two optional text sidecars",
    )
    args = parser.parse_args(argv)
    if args.collection:
        if args.profile != DEFAULT_PROFILE:
            parser.error("--collection uses the predefined independent-text profile")
        profile = rebuild_profile(args.collection)
        with tempfile.TemporaryDirectory(prefix="qwen-text-cold-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile, ensure_ascii=False) + "\n")
            assets = materialize_text_cold_profile(
                args.output_dir, path, overwrite=args.overwrite
            )
    else:
        assets = materialize_text_cold_profile(
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
