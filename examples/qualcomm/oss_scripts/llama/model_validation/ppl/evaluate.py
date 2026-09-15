"""Prepare a fixed local corpus or evaluate teacher-forced standard perplexity."""

import argparse
import contextlib
import fcntl
import importlib.metadata
import math
import os
import platform
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "model_validation.ppl"

from .artifacts import digest, file_hash, read_json, tree_identity, write_json
from .corpus import check_tokenizer, load_corpus, prepare

METRIC = dict(
    name="standard_teacher_forced_perplexity",
    formula="exp(sum(token_nll) / target_count)",
    log_base="e",
    reduction="token_weighted",
    targets="each frozen token at positions 1..N-1 exactly once",
    context_len="tokens per window including its final target; model input excludes that target",
    positions="restart at zero in each window",
    add_special_tokens=False,
    chat_template=False,
    repetition_penalty=1.0,
    thinking_processing=False,
    hf_pt2_cross_entropy="float32 logits, float64 host accumulation",
    pte_cross_entropy="float64 logsumexp; uint16 logits use exported scale",
)


def record_values(record, window):
    """Validate all cached NLLs and their exact alignment, including driver output."""
    values = record.get("token_nll")
    count = window["end"] - window["target_begin"]
    if (
        record.get("window") != window
        or type(record.get("count")) is not int
        or record["count"] != count
        or not isinstance(values, list)
        or len(values) != count
        or not all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values
        )
    ):
        raise ValueError(f"Invalid completed window/NLL values: {window['id']}")
    total = record.get("nll_sum")
    if (
        type(total) not in (int, float)
        or not math.isfinite(total)
        or not math.isclose(math.fsum(values), total, rel_tol=1e-10, abs_tol=1e-10)
    ):
        raise ValueError(f"NLL checksum mismatch: {window['id']}")
    return values


def save_record(output, window, values, identity_hash, details=None):
    """Seal a window with its run identity and a hash covering every NLL."""
    record = dict(
        window=window,
        count=len(values),
        token_nll=values,
        nll_sum=math.fsum(values),
        identity_sha256=identity_hash,
        details=details or {},
    )
    record_values(record, window)
    record["record_sha256"] = digest(record)
    write_json(output / (window["id"] + ".json"), record)
    return record


def summary(identity, records, windows):
    """Aggregate per-token loss with comparable corpus and target fingerprints."""
    for window in windows:
        record = records[window["id"]]
        record_values(record, window)
        if identity["engine"] == "pt2":
            check = record.get("details", {}).get("causality", {})
            if (
                check.get("passed") is not True
                or check.get("policy") != "bit_exact_prefix_logits"
                or check.get("prefix_length") != window["end"] - window["begin"] - 1
                or not check.get("checks")
                or not all(c.get("bit_exact") is True for c in check["checks"])
            ):
                raise ValueError("PT2 result lacks passing causality evidence")
    total = math.fsum(v for w in windows for v in records[w["id"]]["token_nll"])
    count = sum(records[w["id"]]["count"] for w in windows)
    if count != identity["target_count"]:
        raise ValueError("Missing or duplicate target tokens")
    mean = total / count
    ppl = math.exp(mean) if mean <= math.log(sys.float_info.max) else None
    return dict(
        schema="executorch.ppl.summary.v1",
        complete=True,
        standard_ppl_eligible=True,
        engine=identity["engine"],
        nll_sum=total,
        count=count,
        target_count=count,
        mean_nll=mean,
        ppl=ppl,
        ppl_overflow=ppl is None,
        metric=METRIC,
        identity_sha256=digest(identity),
        model_sha256=identity["model"]["sha256"],
        corpus_sha256=identity["corpus_sha256"],
        tokens_sha256=identity["tokens_sha256"],
        targets_sha256=identity["targets_sha256"],
        tokenizer_sha256=identity["tokenizer_sha256"],
        context_len=identity["context_len"],
        stride=identity["stride"],
        record_hashes={w["id"]: records[w["id"]]["record_sha256"] for w in windows},
    )


@contextlib.contextmanager
def locked_output(output, resume):
    """Exclude concurrent writers and refuse accidental reuse without --resume."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another PPL process owns this output directory"
            ) from error
        occupied = any(p.name != ".lock" for p in output.iterdir())
        if occupied and not resume:
            raise ValueError(
                "Run output is not empty; use --resume for the identical run"
            )
        if occupied and not (output / "metadata.json").is_file():
            raise ValueError("Cannot resume an output without run metadata")
        yield


def cached_records(output, identity, windows):
    """Check identity, full records and any completed summary before inference."""
    metadata = output / "metadata.json"
    if metadata.exists() and read_json(metadata) != identity:
        raise ValueError(
            "PPL resume identity mismatch (model/corpus/options/code/runtime changed)"
        )
    expected = {w["id"] + ".json" for w in windows}
    if any(p.name not in expected for p in output.glob("W*.json")):
        raise ValueError("Unexpected cached window")
    records = {}
    for window in windows:
        path = output / (window["id"] + ".json")
        if not path.exists():
            continue
        record = read_json(path)
        record_values(record, window)
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("identity_sha256") != digest(identity) or record.get(
            "record_sha256"
        ) != digest(body):
            raise ValueError(f"Cached record hash/identity mismatch: {window['id']}")
        records[window["id"]] = record
    saved_summary = output / "summary.json"
    if saved_summary.exists():
        if len(records) != len(windows) or read_json(saved_summary) != summary(
            identity, records, windows
        ):
            raise ValueError("Completed summary or record hashes changed")
    return records


def torch_runtime(device):
    """Configure repeatable evaluation and record the actual selected device."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("--device supports cpu or cuda[:index]")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    details = dict(
        torch=torch.__version__,
        device=str(selected),
        tf32=False,
        deterministic=True,
        threads=torch.get_num_threads(),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
        details.update(
            device=str(torch.device("cuda", torch.cuda.current_device())),
            device_name=torch.cuda.get_device_name(selected),
            capability=list(torch.cuda.get_device_capability(selected)),
        )
    else:
        details["processor"] = platform.processor()
    return details


def source_identity():
    """Bind runtime implementation files, excluding documentation and tests."""
    root = Path(__file__).resolve().parent.parent
    sources = [
        p
        for folder in ("ppl", "engines")
        for p in (root / folder).iterdir()
        if p.suffix in (".py", ".cpp", ".h") and not p.name.startswith("test_")
    ]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(sources)}


def score_window(engine, tokens, window, max_token_id=0):
    """Score shifted targets in bounded CE chunks, without sampling."""
    import torch
    import torch.nn.functional as F

    begin, end, target = window["begin"], window["end"], window["target_begin"]
    with torch.inference_mode():
        logits = engine(tokens[begin : end - 1])
        if (
            logits.ndim != 2
            or logits.shape[0] != end - begin - 1
            or not logits.is_floating_point()
            or logits.shape[1] <= max(int(tokens[begin:end].max()), max_token_id)
        ):
            raise ValueError(
                "Engine logits shape/vocabulary does not match fixed inputs"
            )
        labels = torch.as_tensor(tokens[begin + 1 : end], device=logits.device)
        values = []
        for offset in range(target - begin - 1, len(labels), 64):
            chunk = logits[offset : offset + 64].float()
            if not torch.isfinite(chunk).all():
                raise ValueError("Nonfinite model logits")
            values.extend(
                F.cross_entropy(chunk, labels[offset : offset + 64], reduction="none")
                .double()
                .cpu()
                .tolist()
            )
    return values


def run(args):
    """Validate provenance, evaluate missing windows and publish a sealed summary."""
    corpus_root, output, model = (
        args.corpus.resolve(),
        args.output.resolve(),
        args.model.resolve(),
    )
    manifest, tokens, windows = load_corpus(corpus_root)
    tokenizer_path = (
        args.tokenizer.resolve()
        if args.tokenizer
        else (model if args.engine == "hf" else corpus_root / "tokenizer")
    )
    if args.engine == "pte" and tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if args.engine != "pte" and not tokenizer_path.is_dir():
        raise ValueError("HF/PT2 --tokenizer must be a local directory")
    for source in (corpus_root, model, tokenizer_path):
        if output == source or (source.is_dir() and source in output.parents):
            raise ValueError("Run output must be outside corpus/model/tokenizer inputs")
    tokenizer_check = check_tokenizer(tokenizer_path, corpus_root, manifest, tokens)
    model_directory = model if model.is_dir() else model.parent
    sibling_tokenizer = model_directory / "tokenizer.json"
    model_tokenizer_check = None
    has_hf_tokenizer = args.engine == "hf" and any(
        (model_directory / name).is_file()
        for name in (
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "vocab.txt",
        )
    )
    if sibling_tokenizer.is_file() or has_hf_tokenizer:
        model_tokenizer_check = check_tokenizer(
            model_directory if args.engine == "hf" else sibling_tokenizer,
            corpus_root,
            manifest,
            tokens,
        )
    overrides = read_json(args.config_override) if args.config_override else None
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("--config-override must contain a JSON object")
    if args.engine != "hf" and overrides is not None:
        raise ValueError("--config-override is HF-only")
    if args.engine == "pt2" and (
        args.pt2_layout != "lincal-full-sequence" or args.dtype != "float32"
    ):
        raise ValueError(
            "PT2 requires explicit --pt2-layout lincal-full-sequence --dtype float32"
        )
    identity = dict(
        schema="executorch.ppl.run.v1",
        engine=args.engine,
        model=tree_identity(model),
        corpus_sha256=file_hash(corpus_root / "manifest.json"),
        tokens_sha256=manifest["files"]["tokens.u64"],
        targets_sha256=digest(windows),
        tokenizer_sha256=digest(manifest["tokenizer"]),
        tokenizer_check=tokenizer_check,
        model_tokenizer_check=model_tokenizer_check,
        model_config_sha256=(
            file_hash(model_directory / "config.json")
            if (model_directory / "config.json").is_file()
            else None
        ),
        target_count=manifest["target_count"],
        context_len=manifest["context_len"],
        stride=manifest["stride"],
        metric=METRIC,
        implementation=source_identity(),
        python=platform.python_version(),
        config_override=overrides,
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "transformers", "tokenizers")
        },
    )
    pte = None
    if args.engine == "pte":
        from ..engines.pte import PTEEngine

        pte = PTEEngine(args, model, tokenizer_path, manifest, identity)
        identity["runtime"] = pte.identity(offline=args.dry_run)
        if args.dry_run:
            if output.exists() and any(output.iterdir()):
                raise ValueError(
                    "Dry-run output must be empty; use a separate directory"
                )
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "plan.json", pte.plan(identity, windows))
            write_json(
                output / "summary.json",
                dict(
                    complete=False,
                    standard_ppl_eligible=False,
                    status="planned_only",
                    engine="pte",
                    ppl=None,
                    count=0,
                    metric=METRIC,
                    corpus_sha256=identity["corpus_sha256"],
                    tokens_sha256=identity["tokens_sha256"],
                    targets_sha256=identity["targets_sha256"],
                    tokenizer_sha256=identity["tokenizer_sha256"],
                    target_count=identity["target_count"],
                    identity_sha256=digest(identity),
                ),
            )
            return
    else:
        if args.dry_run:
            raise ValueError("--dry-run is PTE-only")
        identity["runtime"] = torch_runtime(args.device)
        identity["dtype"] = args.dtype
        if args.engine == "pt2":
            identity.update(pt2_layout=args.pt2_layout, pad_token_id=args.pad_token_id)
    with locked_output(output, args.resume):
        records = cached_records(output, identity, windows)
        pending = [w for w in windows if w["id"] not in records]
        if not pending:
            if not (output / "summary.json").exists():
                write_json(output / "summary.json", summary(identity, records, windows))
            return
        if not (output / "metadata.json").exists():
            write_json(output / "metadata.json", identity)
        if pte is not None:
            results, details, failure = pte.run(corpus_root, output, identity, pending)
            for window in pending:
                if window["id"] not in results:
                    continue
                values = record_values(results[window["id"]], window)
                records[window["id"]] = save_record(
                    output, window, values, digest(identity), details
                )
            if failure:
                raise RuntimeError(
                    f"PTE driver failed; validated completed windows are resumable: {failure}"
                )
        else:
            if args.engine == "hf":
                from ..engines.hf import HFEngine

                engine = HFEngine(
                    model,
                    device=args.device,
                    dtype=args.dtype,
                    config_override=overrides,
                )
                if manifest["tokenizer"]["max_token_id"] >= engine.vocab_size:
                    raise ValueError("Tokenizer vocabulary exceeds HF input embeddings")
                write_json(output / "engine.json", engine.details)
            else:
                from ..engines.pt2 import PT2Engine

                engine = PT2Engine(
                    model,
                    device=args.device,
                    dtype=args.dtype,
                    layout=args.pt2_layout,
                    pad_token_id=args.pad_token_id,
                )
                if manifest["context_len"] - 1 > engine.max_input_length:
                    raise ValueError(
                        "Corpus context-len exceeds exported PT2 input capacity"
                    )
                write_json(
                    output / "engine.json",
                    dict(
                        input_specs=engine.input_specs,
                        max_input_length=engine.max_input_length,
                        sliding_window=engine.sliding_window,
                        graph_modified=False,
                    ),
                )
            for window in pending:
                started = time.perf_counter()
                values = score_window(
                    engine, tokens, window, manifest["tokenizer"]["max_token_id"]
                )
                details = dict(seconds=time.perf_counter() - started)
                if args.engine == "pt2":
                    details["causality"] = engine.last_causality_check
                records[window["id"]] = save_record(
                    output, window, values, digest(identity), details
                )
                print(
                    f"PPL {args.engine} {window['id']} targets={len(values)}",
                    flush=True,
                )
        if len(records) != len(windows):
            raise ValueError("PTE driver did not return every requested window")
        write_json(output / "summary.json", summary(identity, records, windows))


def parser():
    """Build a self-contained parser suitable for verbatim run.py forwarding."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser(
        "prepare", help="freeze a local tokenizer and UTF-8 text"
    )
    preparation.add_argument("--tokenizer", type=Path, required=True)
    preparation.add_argument("--text", type=Path, required=True)
    preparation.add_argument("--context-len", type=int, required=True)
    preparation.add_argument("--stride", type=int, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser(
        "run", help="evaluate the same fixed tokens on a local model"
    )
    evaluation.add_argument("--engine", choices=("hf", "pt2", "pte"), required=True)
    evaluation.add_argument("--corpus", type=Path, required=True)
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--tokenizer", type=Path)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    evaluation.add_argument(
        "--config-override",
        type=Path,
        help="explicit HF config overrides as a JSON object",
    )
    evaluation.add_argument(
        "--pt2-layout",
        choices=("lincal-full-sequence",),
        default="lincal-full-sequence",
    )
    evaluation.add_argument("--pad-token-id", type=int, default=0)
    evaluation.add_argument("--resume", action="store_true")
    evaluation.add_argument("--serial")
    evaluation.add_argument("--device-lib-dir")
    evaluation.add_argument("--driver", type=Path)
    evaluation.add_argument("--device-model")
    evaluation.add_argument("--mode", choices=("hybrid", "kv"), default="hybrid")
    evaluation.add_argument(
        "--decoder", choices=("lincal", "qwen3", "qwen2_5", "llama3", "llama2")
    )
    evaluation.add_argument(
        "--dry-run",
        action="store_true",
        help="PTE: generate a local plan without any ADB call",
    )
    return result


def main(argv=None):
    """Run either subcommand without interpreting arguments in the parent CLI."""
    arguments = parser().parse_args(argv)
    if arguments.action == "prepare":
        result = prepare(
            arguments.tokenizer,
            arguments.text,
            arguments.context_len,
            arguments.stride,
            arguments.output,
        )
        print(
            f"PREPARED tokens={result['token_count']} targets={result['target_count']}"
        )
    else:
        run(arguments)


if __name__ == "__main__":
    main()
