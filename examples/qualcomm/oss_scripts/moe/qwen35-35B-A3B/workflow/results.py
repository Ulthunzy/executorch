import json
import time


def stream_tokens(device, command, run_dir, tokenizer, text_only):
    generated, timings = [], []
    displayed, previous = "", None
    started = time.perf_counter()
    with (run_dir / "runner.log").open("w") as log:
        with device.stream(command) as process:
            for line in process.stdout:
                arrival = time.perf_counter()
                log.write(line)
                if line.startswith(("PREFILL_TOKEN ", "TOKEN ")):
                    event = json.loads(line.split(" ", 1)[1])
                    if event["index"] == 0 and generated:
                        if event["id"] != generated[0]:
                            raise RuntimeError("Handoff first token mismatch")
                        continue
                    if event["index"] != len(generated):
                        raise RuntimeError("Token stream order mismatch")
                    generated.append(event["id"])
                    row = dict(
                        event,
                        host_ms=(arrival - started) * 1000,
                        host_gap_ms=(
                            None if previous is None else (arrival - previous) * 1000
                        ),
                    )
                    timings.append(row)
                    previous = arrival
                    text = tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).rstrip("\ufffd")
                    if not text.startswith(displayed):
                        raise RuntimeError("Decoded stream prefix changed")
                    delta, displayed = text[len(displayed) :], text
                    if text_only:
                        print(delta, end="", flush=True)
                    else:
                        print(
                            f"TOKEN {event['index']:03d} gap_ms={row['host_gap_ms']} "
                            + json.dumps(delta, ensure_ascii=False),
                            flush=True,
                        )
                elif line.startswith(
                    (
                        "PTE_RESULT ",
                        "VISION_ENCODE ",
                        "MODEL_RESIDENT ",
                        "PREFILL_TOTAL ",
                        "PTE_READY ",
                        "STARTUP_COLD_PRELOAD ",
                        "DECODE_READY ",
                        "EXPERT_HANDOFF_",
                        "EXPERT_HISTORY_",
                        "PREFILL_REUSE_TOTAL ",
                        "EXPERT_RESIDENT_",
                        "EXPERT_CACHE_READY ",
                        "EXPERT_IO_TOTAL ",
                        "DECODE_RESULT ",
                        "DECODE_LOOP ",
                        "DECODE_STOP ",
                        "FATAL ",
                    )
                ):
                    print(line.rstrip(), flush=True)
            exit_code = process.wait()
    return exit_code, generated, timings


def collect_results(args, device, paths, run_dir, vision):
    adb = device.adb
    prefill_root, decode_root = paths.prefill, paths.decode
    if vision is not None:
        adb("pull", prefill_root + "/image-features.f32", run_dir)
    decode_files = [
        "input.i64",
        "output.i64",
        "hidden.f32",
        "logits.f32",
        *[f"layer{i}-cache{c}.u16" for i in range(40) for c in range(2)],
    ]
    adb(
        "pull",
        *[decode_root + "/decode-pte-0-" + suffix for suffix in decode_files],
        run_dir,
    )
    for suffix in decode_files:
        (run_dir / ("decode-pte-0-" + suffix)).rename(run_dir / suffix)
    history_files = [
        "prefill-history-ids.i32",
        "prefill-history-stamps.u64",
        "prefill-routes.i32",
        "prefill-all-routes.i32",
        "prefill-chunk-seeds.i32",
        "prefill-chunk-stamps.u64",
    ]
    adb("pull", *[prefill_root + "/" + name for name in history_files], run_dir)
    adb(
        "pull",
        *[
            decode_root + "/" + name
            for name in [
                "cache-final-ids.i32",
                "cache-seed-ids.i32",
                "reused-ids.i32",
                "reused-stamps.u64",
                "hot-masks.u32",
            ]
        ],
        run_dir,
    )
    (run_dir / "prefill").mkdir()
    prefill_files = [
        "hidden.f32",
        "logits.f32",
        *[f"layer{i}-cache{c}.u16" for i in range(40) for c in range(2)],
    ]
    adb(
        "pull",
        *[prefill_root + "/pte-0-" + suffix for suffix in prefill_files],
        run_dir / "prefill",
    )
    for suffix in prefill_files:
        (run_dir / "prefill" / ("pte-0-" + suffix)).rename(run_dir / "prefill" / suffix)


def verify_results(args, run_dir, tokenizer, generated, teacher, eos_ids):
    import numpy as np

    inputs = np.fromfile(run_dir / "input.i64", dtype="<i8")
    outputs = np.fromfile(run_dir / "output.i64", dtype="<i8")
    if (
        len(inputs) != len(outputs)
        or len(inputs) != len(generated) - 1
        or generated[1:] != outputs.tolist()
    ):
        raise RuntimeError("Generated token counts disagree with device outputs")
    if args.teacher_tokens:
        if not np.array_equal(inputs, teacher[: len(inputs)]):
            raise RuntimeError("Teacher forcing input mismatch")
    elif len(inputs) and (
        inputs[0] != generated[0] or not np.array_equal(inputs[1:], outputs[:-1])
    ):
        raise RuntimeError("Greedy decode feedback mismatch")
    end = next(
        (i for i, token in enumerate(generated) if token in eos_ids),
        None,
    )
    if not args.ignore_eos and end is not None and end != len(generated) - 1:
        raise RuntimeError("Generation continued after EOS")
    answer = dict(
        answer=tokenizer.decode(
            generated if end is None else generated[:end],
            skip_special_tokens=True,
        ),
        input_ids=inputs.tolist(),
        output_ids=outputs.tolist(),
        eos_index=end,
        greedy_feedback_verified=not bool(args.teacher_tokens),
        teacher_forcing_verified=bool(args.teacher_tokens),
        generated_tokens=len(generated),
    )
    (run_dir / "ANSWER.json").write_text(
        json.dumps(answer, ensure_ascii=False, indent=2) + "\n"
    )
    print("\nANSWER " + answer["answer"], flush=True)
    print("SAVED", run_dir, flush=True)
    return answer
