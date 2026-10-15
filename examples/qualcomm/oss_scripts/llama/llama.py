# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import argparse
import json
import logging
import math
import os
import shutil
import sys
from multiprocessing.connection import Client
from typing import Dict, List

import torch
from executorch.backends.qualcomm.debugger.utils import QnnTool
from executorch.backends.qualcomm.export_utils import (
    get_backend_type,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.utils.check_qnn_version import describe_sdk_build_id

from executorch.backends.qualcomm.utils.utils import (
    dump_context_from_pte,
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLMModelConfig,
    SUPPORTED_LLM_MODELS,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset import (
    DataConfig,
    DatasetBuilder,
    MessageSample,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    ATTENTION_SINK_EVICTOR,
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    EVAL_MODE,
    PROMPT_EVAL,
    SQNR_EVAL,
    TASKS_EVAL,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    TOK_EMBEDDING_GRAPH_NAMES,
    VISION_ENCODER,
)


def _ensure_transformers_vlm_eval_compat():
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
        transformers, "AutoModelForImageTextToText"
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText


_ensure_transformers_vlm_eval_compat()

from executorch.examples.qualcomm.oss_scripts.llama.evaluator.device_evaluator import (
    DefaultEval,
    SqnrEval,
    TaskEval,
)

from executorch.examples.qualcomm.oss_scripts.llama.gdn import (
    DEFAULT_OP_PACKAGE_DIR,
    is_qwen35_model,
    qwen35_prefill_capacity,
    setup_gdn_op_package,
)
from executorch.examples.qualcomm.oss_scripts.llama.model_validation.common.lineage import (
    write_export_lineage,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import (
    HybridAttentionSinkEvictor,
    is_attention_sink_config_equal,
    MultiModalManager,
    next_power_of_two,
)
from torchao.quantization.utils import compute_error


sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)
# Avoid the error message "Could not initialize NNPACK! Reason: Unsupported hardware."
torch.backends.nnpack.set_flags(False)


def _dumpfile_to_method_name(dump_file: str) -> str:
    stem = os.path.splitext(os.path.basename(dump_file))[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _get_qnn_dummy_tensor(shape, data_type, qnn_data_type):
    dtype_map = {
        qnn_data_type.QNN_DATATYPE_BOOL_8: torch.bool,
        qnn_data_type.QNN_DATATYPE_FLOAT_16: torch.float16,
        qnn_data_type.QNN_DATATYPE_FLOAT_32: torch.float32,
        qnn_data_type.QNN_DATATYPE_INT_8: torch.int8,
        qnn_data_type.QNN_DATATYPE_INT_16: torch.int16,
        qnn_data_type.QNN_DATATYPE_INT_32: torch.int32,
        qnn_data_type.QNN_DATATYPE_INT_64: torch.int64,
        qnn_data_type.QNN_DATATYPE_SFIXED_POINT_8: torch.int8,
        qnn_data_type.QNN_DATATYPE_SFIXED_POINT_16: torch.int16,
        qnn_data_type.QNN_DATATYPE_SFIXED_POINT_32: torch.int32,
        qnn_data_type.QNN_DATATYPE_UINT_8: torch.uint8,
        qnn_data_type.QNN_DATATYPE_UINT_16: torch.uint16,
        qnn_data_type.QNN_DATATYPE_UINT_32: torch.int32,
        qnn_data_type.QNN_DATATYPE_UFIXED_POINT_8: torch.uint8,
        qnn_data_type.QNN_DATATYPE_UFIXED_POINT_16: torch.uint16,
        qnn_data_type.QNN_DATATYPE_UFIXED_POINT_32: torch.int32,
    }
    dtype = dtype_map.get(data_type, torch.float32)
    return torch.zeros(shape, dtype=dtype)


def _build_qnn_profile_inputs(args, dump_file: str, fallback_inputs):
    if os.path.splitext(dump_file)[1] != ".bin":
        return fallback_inputs

    try:
        import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor
        from executorch.backends.qualcomm.partition.utils import (
            generate_qnn_executorch_option,
        )
        from executorch.examples.qualcomm.qaihub_scripts.utils.utils import (
            preprocess_binary,
        )
    except Exception as exc:
        logging.warning(
            "Falling back to exported example inputs for %s because QNN pybind helpers are unavailable: %s",
            dump_file,
            exc,
        )
        return fallback_inputs

    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.model],
        backend_options=backend_options,
        is_from_context_binary=True,
    )

    with open(dump_file, "rb") as f:
        ctx_bin = preprocess_binary(f.read(), compiler_specs)

    qnn_mgr = PyQnnManagerAdaptor.QnnManager(
        generate_qnn_executorch_option(compiler_specs), ctx_bin
    )
    try:
        err = qnn_mgr.Init().value
        if err != 0:
            raise RuntimeError(f"failed to load context binary, error code {err}")

        graph_name = qnn_mgr.GetGraphNames()[0]
        qnn_mgr.AllocateTensor(graph_name)
        qnn_inputs = []
        for tensor in qnn_mgr.GetGraphInputs(graph_name):
            dims = tensor.GetDims()
            shape = tuple(int(dim) for dim in dims.tolist())
            qnn_inputs.append(
                _get_qnn_dummy_tensor(
                    shape,
                    tensor.GetDataType(),
                    PyQnnManagerAdaptor.Qnn_DataType_t,
                )
            )
        return [tuple(qnn_inputs)]
    except Exception as exc:
        logging.warning(
            "Falling back to exported example inputs for %s because QNN graph inputs could not be introspected: %s",
            dump_file,
            exc,
        )
        return fallback_inputs
    finally:
        qnn_mgr.Destroy()


def _build_optrace_targets(
    args,
    multi_modal_mgr,
    pte_filenames: Dict[str, str],
):
    pte_targets = {}

    decoder_inputs = {}
    if multi_modal_mgr.text_decoder.decode.decoder is not None:
        decoder_inputs["kv_forward"] = [
            multi_modal_mgr.text_decoder.decode.export_input
        ]
    if multi_modal_mgr.text_decoder.prefill.decoder is not None:
        decoder_inputs["prefill_forward"] = [
            multi_modal_mgr.text_decoder.prefill.export_input
        ]
    for name, bucket in multi_modal_mgr.text_decoder.prefill_buckets:
        if bucket.decoder is not None:
            decoder_inputs[name] = [bucket.export_input]
    if decoder_inputs:
        pte_targets[f"{args.artifact}/{pte_filenames[TEXT_DECODER]}.pte"] = (
            decoder_inputs
        )

    if multi_modal_mgr.text_decoder.apply_embedding:
        embedding_inputs = {}
        if multi_modal_mgr.text_decoder.decode.tok_embedding is not None:
            embedding_inputs["tok_embedding_kv_forward"] = [
                multi_modal_mgr.text_decoder.decode.tok_embedding_export_input
            ]
        if multi_modal_mgr.text_decoder.prefill.tok_embedding is not None:
            embedding_inputs["tok_embedding_prefill_forward"] = [
                multi_modal_mgr.text_decoder.prefill.tok_embedding_export_input
            ]
        if embedding_inputs:
            pte_targets[f"{args.artifact}/{pte_filenames[TOK_EMBEDDING]}.pte"] = (
                embedding_inputs
            )

    for modality_name in (VISION_ENCODER, TEXT_ENCODER, AUDIO_ENCODER):
        modality = getattr(multi_modal_mgr, modality_name, None)
        if modality is None or modality.model is None:
            continue
        pte_targets[f"{args.artifact}/{pte_filenames[modality_name]}.pte"] = {
            "forward": [modality.example_input]
        }

    return pte_targets


def _generate_optrace_reports(
    args,
    multi_modal_mgr,
    pte_filenames: Dict[str, str],
):
    pte_targets = _build_optrace_targets(args, multi_modal_mgr, pte_filenames)
    trace_root = os.path.join(args.artifact, "optrace")
    os.makedirs(trace_root, exist_ok=True)
    manifest = {}

    for pte_path, method_inputs in pte_targets.items():
        if not os.path.exists(pte_path):
            logging.warning("Skipping optrace for missing pte: %s", pte_path)
            continue

        pte_name = os.path.splitext(os.path.basename(pte_path))[0]
        pte_trace_dir = os.path.join(trace_root, pte_name)
        os.makedirs(pte_trace_dir, exist_ok=True)
        manifest[pte_name] = {}

        for dump_file in dump_context_from_pte(pte_path):
            method_name = _dumpfile_to_method_name(dump_file)
            example_inputs = method_inputs.get(method_name)
            if example_inputs is None:
                logging.warning(
                    "Skipping dumped binary %s because no example inputs were found for method %s.",
                    dump_file,
                    method_name,
                )
                continue

            method_trace_dir = os.path.join(pte_trace_dir, method_name)
            os.makedirs(method_trace_dir, exist_ok=True)
            copied_dump = shutil.copy2(dump_file, method_trace_dir)
            workspace = f"/data/local/tmp/executorch/{pte_name}_{method_name}"
            profiling_inputs = _build_qnn_profile_inputs(
                args, copied_dump, example_inputs
            )
            adb = SimpleADB(
                qnn_sdk=os.getenv("QNN_SDK_ROOT"),
                build_path=f"{args.build_folder}",
                pte_path=pte_path,
                workspace=workspace,
                device_id=args.device,
                host_id=args.host,
                soc_model=args.model,
                shared_buffer=args.shared_buffer,
                target=args.target,
            )
            qnn_tool = QnnTool(
                method_trace_dir,
                profiling_inputs,
                get_soc_to_chipset_map()[args.model],
                adb,
                build_folder=args.build_folder,
                workspace=workspace,
            )
            optrace_path, qhas_path = qnn_tool.generate_optrace(
                os.path.basename(copied_dump)
            )
            manifest[pte_name][method_name] = {
                "binary": copied_dump,
                "optrace": optrace_path,
                "qhas": qhas_path,
            }
            logging.info(
                "Generated optrace for %s/%s under %s",
                pte_name,
                method_name,
                method_trace_dir,
            )

        manifest_path = os.path.join(pte_trace_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest[pte_name], f, indent=2, sort_keys=True)

    with open(os.path.join(trace_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def compile_attention_sink_evictor(
    args,
    decoder_model_config: LLMModelConfig,
    text_decoder_pte_path: str,
    attention_sink_evictor_pte_path: str,
):
    # For inference, we will check the attention sink configurations to determine if recompilation is needed.
    # For compilation, the attention sink evictor will always be recompiled.
    if (
        args.pre_gen_pte
        and os.path.exists(attention_sink_evictor_pte_path)
        and is_attention_sink_config_equal(attention_sink_evictor_pte_path, args)
    ):
        logging.info("Attention sink evictor is already compiled, skipping...")
        return
    attention_sink_evictor = HybridAttentionSinkEvictor(
        control_args=args, config=decoder_model_config
    )
    attention_sink_evictor.quantize(text_decoder_pte_path)
    attention_sink_evictor.compile(attention_sink_evictor_pte_path)


def compile(
    args,
    decoder_model_config: LLMModelConfig,
    pte_filenames: Dict[str, str],
    tokenizer_wrapper,
    is_multimodal,
) -> Dict[str, List]:
    os.makedirs(args.artifact, exist_ok=True)
    op_package_options = None
    if is_qwen35_model(args.decoder_model):
        op_package_options, _ = setup_gdn_op_package(args)
        logging.info("Using DeltaNetCoreVtcm with calibrated UINT16 state")
    multi_modal_mgr = MultiModalManager(control_args=args, config=decoder_model_config)

    skip_quantize = {}

    # Prepare ptq option and compile spec
    compile_specs = {
        AUDIO_ENCODER: None,
        TEXT_ENCODER: None,
        VISION_ENCODER: None,
        TOK_EMBEDDING: None,
        TEXT_DECODER: None,
    }
    use_weight_sharing = decoder_model_config.use_weight_sharing
    for modality in compile_specs:
        if is_multimodal and modality in {AUDIO_ENCODER, TEXT_ENCODER, VISION_ENCODER}:
            to_skip = (
                modality == VISION_ENCODER
                and decoder_model_config.vision_encoder is not None
                and decoder_model_config.vision_encoder.use_fp16
            )
            if args.backend == "htp":
                backend_options = generate_htp_compiler_spec(
                    use_fp16=to_skip,
                )
            elif args.backend == "gpu":
                backend_options = generate_gpu_compiler_spec()
            else:
                raise ValueError(f"Unsupported backend {args.backend}")
            encoder_compile_specs = generate_qnn_executorch_compiler_spec(
                soc_model=get_soc_to_chipset_map()[args.soc_model],
                backend_options=backend_options,
                online_prepare=args.online_prepare,
                profile_level=3 if args.optrace else 0,
                dump_intermediate_outputs=args.dump_intermediate_outputs,
                # x86 emulator does not support shared buffer
                shared_buffer=not args.enable_x86_64,
            )
            skip_quantize[modality] = to_skip
            compile_specs[modality] = encoder_compile_specs
        elif is_multimodal and modality == TOK_EMBEDDING:
            if args.backend == "htp":
                backend_options = generate_htp_compiler_spec(
                    use_fp16=False,
                    use_weight_sharing=use_weight_sharing,
                )
            elif args.backend == "gpu":
                backend_options = generate_gpu_compiler_spec()
            else:
                raise ValueError(f"Unsupported backend {args.backend}")
            compile_specs[modality] = [
                generate_qnn_executorch_compiler_spec(
                    soc_model=get_soc_to_chipset_map()[args.soc_model],
                    backend_options=backend_options,
                    online_prepare=args.online_prepare,
                    profile_level=3 if args.optrace else 0,
                    dump_intermediate_outputs=args.dump_intermediate_outputs,
                    # x86 emulator does not support shared buffer
                    shared_buffer=not args.enable_x86_64,
                )
            ] * len(TOK_EMBEDDING_GRAPH_NAMES)
        elif modality == TEXT_DECODER:
            # compile spec for text decoder
            if args.backend == "htp":
                backend_options = generate_htp_compiler_spec(
                    use_fp16=args.use_fp16,
                    use_multi_contexts=decoder_model_config.num_sharding > 1,
                    use_weight_sharing=use_weight_sharing,
                )
            elif args.backend == "gpu":
                backend_options = generate_gpu_compiler_spec()
            else:
                raise ValueError(f"Unsupported backend {args.backend}")
            skip_quantize[modality] = args.use_fp16
            compile_specs[modality] = [
                generate_qnn_executorch_compiler_spec(
                    soc_model=get_soc_to_chipset_map()[args.soc_model],
                    backend_options=backend_options,
                    online_prepare=args.online_prepare,
                    profile_level=3 if args.optrace else 0,
                    dump_intermediate_outputs=args.dump_intermediate_outputs,
                    # x86 emulator does not support shared buffer
                    shared_buffer=not args.enable_x86_64,
                    use_mha2sha=not args.disable_mha2sha
                    and not is_qwen35_model(args.decoder_model),
                    op_package_options=op_package_options,
                )
            ] * len(DECODER_GRAPH_NAMES)

    # perform ptq
    multi_modal_mgr.quantize(
        tokenizer_wrapper=tokenizer_wrapper,
        skip_quantize=skip_quantize,
        backend=get_backend_type(args.backend),
        soc_model=args.soc_model,
    )

    # perform compilation
    multi_modal_mgr.compile(
        compile_specs=compile_specs,
        pte_filenames=pte_filenames,
        skip_quantize=skip_quantize,
    )
    if args.optrace:
        _generate_optrace_reports(args, multi_modal_mgr, pte_filenames)


def _file_state(path):
    if not os.path.isfile(path):
        return None
    stat = os.stat(path)
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def inference(
    args,
    decoder_model_config: LLMModelConfig,
    runtime_tokenizer_path,
    tokenizer_wrapper: TokenizerWrapper,
    text_decoder_pte_path: str,
    encoder_pte_paths: Dict[str, str],
    tok_embedding_pte_path: str,
    attention_sink_evictor_pte_path: str,
    is_multimodal,
):
    tokenizer = tokenizer_wrapper.tokenizer

    assert args.model_mode in EVAL_MODE, f"Unknown model_mode: {args.model_mode}."

    pte_paths = {TEXT_DECODER: text_decoder_pte_path}
    _validate_pte_paths(pte_paths)
    eval_results = {
        "pte_size": os.path.getsize(text_decoder_pte_path),
    }

    if args.use_attention_sink:
        pte_paths.update({ATTENTION_SINK_EVICTOR: attention_sink_evictor_pte_path})
        _validate_pte_paths({ATTENTION_SINK_EVICTOR: attention_sink_evictor_pte_path})
        eval_results.update(
            {
                "attention_sink_evictor_pte_size": os.path.getsize(
                    attention_sink_evictor_pte_path
                ),
            }
        )

    if is_multimodal:
        required_multimodal_pte_paths = {TOK_EMBEDDING: tok_embedding_pte_path}
        required_multimodal_pte_paths.update(encoder_pte_paths)
        _validate_pte_paths(required_multimodal_pte_paths)
        eval_results.update(
            {
                "tok_embedding_pte_size": os.path.getsize(tok_embedding_pte_path),
            }
        )
        pte_paths.update(
            {
                TOK_EMBEDDING: tok_embedding_pte_path,
            }
        )
        for modality, encoder_pte_path in encoder_pte_paths.items():
            eval_results.update(
                {f"{modality}_pte_size": os.path.getsize(encoder_pte_path)}
            )
            pte_paths.update(
                {modality: encoder_pte_path},
            )

    multi_modal_mgr = MultiModalManager(control_args=args, config=decoder_model_config)
    audio_encoder = multi_modal_mgr.audio_encoder.model
    vision_encoder = multi_modal_mgr.vision_encoder.model
    tok_embedding = multi_modal_mgr.text_decoder.calibration_prefill.tok_embedding
    source_model = multi_modal_mgr.text_decoder.calibration_prefill.decoder
    audio_token_id = multi_modal_mgr.text_decoder.calibration_prefill.meta.get(
        "audio_token_id", None
    )
    image_token_id = multi_modal_mgr.text_decoder.calibration_prefill.meta.get(
        "image_token_id", None
    )
    dataset_builder = DatasetBuilder(
        DataConfig.from_args(args),
        decoder_model_config,
        tokenizer_wrapper,
        attn_mask=source_model.get_example_inputs()[1],
    )
    if PROMPT_EVAL in args.eval_methods:
        prompt_evaluator = DefaultEval(
            args=args,
            decoder_model_config=decoder_model_config,
            pte_paths=pte_paths,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_multimodal=is_multimodal,
            dataset_builder=dataset_builder,
        )
        output_prompt = prompt_evaluator.run(
            prompt=args.prompt, audio_paths=args.audio_path, image_paths=args.image_path
        )
        eval_results.update(
            {
                "inference_speed": prompt_evaluator.inference_speed,
                "result": output_prompt,
            }
        )
        for idx, output in enumerate(output_prompt):
            logging.info(f"Device Inference Results[{idx}]:\n{output}")

    if SQNR_EVAL in args.eval_methods:
        assert not is_multimodal, "Modality Model does not support SQNR_EVAL."
        runtime_message = tokenizer_wrapper.prepare_messages(args.prompt)[0]
        message = MessageSample(
            files=runtime_message["files_path"],
            messages=tokenizer_wrapper.make_chat_template(
                runtime_message["text"], args.system_prompt
            ),
        )
        sqnr_evaluator = SqnrEval(
            source_model=source_model,
            get_example_inputs=source_model.get_example_inputs,
            args=args,
            pte_paths=pte_paths,
            tokenizer_wrapper=tokenizer_wrapper,
            decoder_model_config=decoder_model_config,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_multimodal=is_multimodal,
            dataset_builder=dataset_builder,
            encoder=audio_encoder or vision_encoder,
            tok_embedding=tok_embedding,
            audio_token_id=audio_token_id,
            image_token_id=image_token_id,
        )
        sqnr, golden_logits, _ = sqnr_evaluator.run(
            message, audio_paths=args.audio_path, image_paths=args.image_path
        )
        logging.info(f"SQNR Eval Score between FP32 nn.Module and QNN: {sqnr}")
        eval_results.update(
            {
                "sqnr": sqnr,
                "inference_speed": sqnr_evaluator.inference_speed,
            }
        )

        qdq_ep_path = (
            f"{args.pre_gen_pte}/{DECODE_QDQ_FILENAME}"
            if args.pre_gen_pte
            else f"{args.artifact}/{DECODE_QDQ_FILENAME}"
        )
        if os.path.exists(qdq_ep_path):
            qdq_ep = torch.export.load(qdq_ep_path)
            qdq_sqnr_evaluator = SqnrEval(
                source_model=qdq_ep.module(),
                get_example_inputs=source_model.get_example_inputs,
                args=args,
                pte_paths=pte_paths,
                tokenizer_wrapper=tokenizer_wrapper,
                decoder_model_config=decoder_model_config,
                runtime_tokenizer_path=runtime_tokenizer_path,
                is_multimodal=is_multimodal,
                dataset_builder=dataset_builder,
                encoder=audio_encoder or vision_encoder,
                tok_embedding=tok_embedding,
                audio_token_id=audio_token_id,
                image_token_id=image_token_id,
            )
            qdq_sqnr, cpu_qdq_logits, _ = qdq_sqnr_evaluator.run(
                message, audio_paths=args.audio_path, image_paths=args.image_path
            )
            eval_results["qdq_sqnr"] = qdq_sqnr
            logging.info(f"SQNR Eval Score between CPU QDQ and QNN: {qdq_sqnr}")
            logging.info(
                f"SQNR Eval Score between FP32 nn.Module and CPU QDQ: {compute_error(golden_logits, cpu_qdq_logits).item()}"
            )
        else:
            logging.info(
                f"Couldn't find saved qdq_ep under {qdq_ep_path}, skip eval sqnr for CPU QDQ."
            )

    if TASKS_EVAL in args.eval_methods:
        assert not is_multimodal, "Multimodal does not support TASKS_EVAL."
        # Generate the eval wrapper
        ppl_evaluator = TaskEval(
            args=args,
            pte_paths=pte_paths,
            tokenizer=tokenizer,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_multimodal=is_multimodal,
        )
        ppl_eval_result = ppl_evaluator.run()
        eval_results["inference_speed"] = ppl_evaluator.inference_speed

        for task, res in ppl_eval_result["results"].items():
            match task:
                case "wikitext":
                    wiki_ppl = ppl_eval_result["results"][task]["word_perplexity,none"]
                    eval_results["wiki_ppl"] = wiki_ppl
                case "hellaswag":
                    acc_norm = ppl_eval_result["results"][task]["acc_norm,none"]
                    eval_results["acc_norm"] = acc_norm
                case _:
                    if args.ip and args.port != -1:
                        raise RuntimeError(
                            "CI currently supports [wikitext, hellaswag] only."
                        )
            logging.info(f"{task}: {res}")

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps(eval_results))


def _normalize_decoder_model_name(name: str) -> str:
    canonical = name.lower().replace(".", "_")
    if canonical in SUPPORTED_LLM_MODELS:
        return canonical
    raise argparse.ArgumentTypeError(
        f"Unknown decoder_model: {name}. Available options are: "
        f"{', '.join(SUPPORTED_LLM_MODELS.keys())}"
    )


def _validate_pte_paths(pte_paths: Dict[str, str]) -> None:
    missing = [
        f"{component}: {path}"
        for component, path in pte_paths.items()
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required pre-generated PTE artifact(s):\n" + "\n".join(missing)
        )


def _repetition_penalty(value):
    penalty = float(value)
    if not math.isfinite(penalty) or penalty < 1.0:
        raise argparse.ArgumentTypeError("repetition_penalty must be finite and >= 1.0")
    return penalty


def _build_parser():
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama_qnn",
        default="./llama_qnn",
        type=str,
    )

    parser.add_argument(
        "--decoder_model",
        choices=list(SUPPORTED_LLM_MODELS.keys()),
        type=_normalize_decoder_model_name,
        help=f"The llm model to export. Current available options are: { SUPPORTED_LLM_MODELS.keys()}",
        required=True,
    )

    parser.add_argument(
        "--gdn_op_package_dir",
        default=str(DEFAULT_OP_PACKAGE_DIR),
        help="DeltaNet VTCM op package source directory for Qwen3.5.",
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--source_safetensors",
        help="Local Hugging Face model directory for weights, tokenizer and encoders. When --checkpoint is given, it remains the decoder weight source.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_bin",
        help="For Llama2. Pass Llama2 tokenizer binary.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts used during runtime inference only (not compilation or calibration). When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.",
        required=True,
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--system_prompt",
        help="For Llama3/Granite. Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None",
        default="",
        type=str,
    )

    parser.add_argument(
        "--temperature",
        help="Sampling temperature for llama.",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "--repetition_penalty",
        type=_repetition_penalty,
        default=1.0,
        help="Runtime repetition penalty over the most recent 256 prompt/generated tokens, including thinking and earlier turns; 1.0 disables it. Does not affect calibration or compilation.",
    )

    parser.add_argument(
        "--ignore_eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore EOS during prompt evaluation and generate to max_seq_len for fixed-length benchmarking. Does not affect compilation or calibration.",
    )
    parser.add_argument(
        "--repair_think_eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="QNN text KV/hybrid only: replace premature EOS inside think with </think>.",
    )
    parser.add_argument(
        "--enable_think_budget",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="QNN text KV/hybrid only: close think at its token budget and continue answering.",
    )
    parser.add_argument(
        "--think_budget",
        type=int,
        default=512,
        help="Positive runtime thinking-token budget; no re-export required.",
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp32",
    )

    parser.add_argument(
        "--model_mode",
        help=f"Export and inference in {EVAL_MODE.keys()}",
        default="hybrid",
        choices=list(EVAL_MODE.keys()),
        type=str,
    )

    parser.add_argument(
        "--enable_true_swa",
        help="Enable true sliding-window attention export when the decoder family supports it.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max_context_len",
        help="Maximum length of the model's memory/cache, including both prompt tokens and generated tokens.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--disable_mha2sha",
        help="Disable the MHA-to-SHA graph rewrite for decoder compilation.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max_seq_len",
        help="Maximum sequence length the model can generate.",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--prefill_ar_len",
        help="The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use this option to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor for hybrid and lookahead mode. Larger values can improve prompt throughput but require more device memory.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="Fallback to cpu embedding operator and type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '4,32'.",
    )

    parser.add_argument(
        "--ngram",
        help="Represents the size of the n-grams used in the lookahead process.",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--window",
        help="Determines how many future tokens the algorithm attempts to predict in each step.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--gcap",
        help="Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--use_attention_sink",
        default=None,
        type=str,
        help="Use the attention sink feature to have fluent multi-round conversations. Specify the settings as '<sink_size>,<batch_eviction_size>', for example, '4,32'."
        "This setting is for compilation. Once you compile with a chosen <sink_size> and <batch_eviction_size>, they cannot be changed at runtime. If you need to update them, you can recompile the attention sink module along with llama.py.",
    )

    parser.add_argument(
        "--audio_path",
        help="Path to the audio file used during runtime inference only (not compilation or calibration). For multimodal language models (MLLM). If not specified, the default audio from encoder/encoder_config.py will be used. The audio should be preprocessed and saved in raw binary format.",
        default=[],
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--image_path",
        help="Path to the image file used during runtime inference only (not compilation or calibration). For multimodal language models (MLLM). If not specified, the default image from encoder/encoder_config.py will be used. The image should be preprocessed and saved in raw binary format.",
        default=[],
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--optrace",
        "--obtrace",
        dest="optrace",
        action="store_true",
        help="Generate QNN optrace/QHAS reports after compilation. This requires an attached device and stores reports under <artifact>/optrace.",
    )

    parser.add_argument(
        "--eval_methods",
        choices=[PROMPT_EVAL, TASKS_EVAL, SQNR_EVAL],
        nargs="+",
        default=[PROMPT_EVAL],
        help="Choose eval methods(default: prompt_eval). Users can provide more than 1 eval methods. For example: --eval_methods tasks_eval sqnr_eval."
        "Following eval methods are supported:"
        "1) prompt_eval: Model will generate the output response based on the provided prompt through the flag --prompt."
        "2) tasks_eval: This will eval the tasks provided through the flag --eval_tasks."
        "3) sqnr_eval: This will eval the sqnr between between QNN's output logit V.S. Static Llama nn.Module's output logit. Eval is based on the provided prompt through the --prompt flag. Please note that sqnr will only eval the prompt's logit but not the new generated token's logit.",
    )

    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        type=str,
        default=None,
        help="list of lm-evaluation-harness tasks to evaluate usage: --eval_tasks task1 task2",
    )

    parser.add_argument(
        "--eval_limit",
        type=int,
        default=1,
        help="number of samples to evalulate. If not set, evaluate all samples",
    )

    parser.add_argument(
        "--eval_num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples to eval in few-shot context",
    )

    parser.add_argument(
        "--calib_tasks",
        nargs="+",
        type=str,
        default=None,
        help="list of lm-evaluation-harness tasks to calibrate usage: --calib_tasks task1 task2",
    )

    parser.add_argument(
        "--calib_limit",
        type=int,
        default=1,
        help="number of samples to calibrate. If not set, calibrate all samples",
    )

    parser.add_argument(
        "--calib_num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples to calibrate in few-shot context",
    )

    parser.add_argument(
        "--calib_samples",
        nargs="+",
        type=str,
        default=None,
        help="One or more paths to calibration sample JSON files. Only JSON format is supported. "
        "Each file must be a flat list of sample objects: "
        '[{"files": ["path_or_url", ...], "messages": [{"role": "user"|"assistant", "content": "..." | [...]}]}]. '
        '"files" is optional and only required for multimodal models (VLM: image paths/URLs, ALM: audio paths/URLs). '
        '"messages" follows the HuggingFace chat template; "content" can be a plain string or a list of content blocks. '
        "Multiple files are merged.",
    )

    parser.add_argument(
        "--calib_hf_dataset",
        type=str,
        default=None,
        help="HuggingFace chat dataset to use as additional calibration data "
        "(e.g. 'HuggingFaceTB/smol-smoltalk').",
    )

    parser.add_argument(
        "--calib_hf_limit",
        type=int,
        default=1,
        help="Number of samples to load from --calib_hf_dataset.",
    )

    parser.add_argument(
        "--calib_text_datasets",
        nargs="+",
        default=None,
        help="Raw-text calibration sources: cci3-hq, wikitext-2, or local JSON/JSONL files with a 'text' field. Hub sources stream from the train split.",
    )

    parser.add_argument(
        "--calib_text_samples",
        nargs="+",
        type=int,
        default=[128],
        help="Number of calibration sequences per text source (default: 128 each). Supply one count for all sources or one per source. Preserved sources use one complete row per sequence.",
    )

    parser.add_argument(
        "--calib_text_preserve_sources",
        nargs="+",
        default=None,
        help="Local JSON/JSONL paths also listed in --calib_text_datasets whose rows must remain intact. No template, packing, truncation or automatic EOS insertion is applied. Oversized rows fail explicitly; other sources retain their existing packing behavior.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for text decoder quantization. Larger values increase throughput "
        "but require more host memory. Only affects the CALIBRATE graph; DECODE and "
        "PREFILL graphs always use batch size 1.",
    )

    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable Quantization-Aware Training (QAT). If not set, defaults to PTQ.",
    )

    parser.add_argument(
        "--train_config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "train", "config", "qad.yaml"),
        help="(QAT) Path to a YAML file overriding TrainingArgs defaults "
        "(train/config/config.py) — e.g. epochs, lr, alpha, temperature, "
        "grad_accum_steps, max_grad_norm, warmup_ratio. Defaults to "
        "train/config/qad.yaml.",
    )

    parser.add_argument(
        "--lr_config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "train", "config", "lr_config.yaml"
        ),
        help="(QAT) Path to a YAML file mapping glob patterns over "
        "model.named_parameters() names to per-param-group optimizer kwargs "
        "(lr, weight_decay, betas, eps, ...). Defaults to "
        "train/config/lr_config.yaml.",
    )

    parser.add_argument(
        "--train_tasks",
        nargs="+",
        type=str,
        default=None,
        help="(QAT) list of lm-evaluation-harness tasks for training data. "
        "Usage: --train_tasks task1 task2."
        "And you need to use --calib_tasks to specify calibration data separately. ",
    )

    parser.add_argument(
        "--train_limit",
        type=int,
        default=1,
        help="(QAT) number of samples for train task limits.",
    )

    parser.add_argument(
        "--train_hf_dataset",
        type=str,
        default=None,
        help="(QAT) HuggingFace instruct dataset for training "
        "(e.g. 'HuggingFaceTB/smol-smoltalk')."
        "And you may need to use --calib_tasks to specify calibration data separately. ",
    )

    parser.add_argument(
        "--train_hf_limit",
        type=int,
        default=1000,
        help="(QAT) Number of samples to load from --train_hf_dataset. "
        "And you may need to use --calib_hf_dataset to specify calibration data separately. ",
    )

    parser.add_argument(
        "--train_val_ratio",
        type=float,
        default=1.0,
        help="(QAT) Fraction of the non-calib samples used for training. "
        "The remaining becomes the validation set. "
        "e.g. 0.9 → 90%% train, 10%% val. 1.0 disables validation.",
    )

    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting",
        action="store_true",
        default=False,
    )

    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument(
        "--calibration_device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for decoder PTQ calibration only. CUDA requires a visible NVIDIA GPU; QNN compilation remains on CPU.",
    )

    parser.add_argument(
        "--skip_qdq_save",
        action="store_true",
        help="Skip the full-sequence QDQ evaluation artifact to reduce host export memory and disk use. Does not skip calibration or change the deployed PTE. Incompatible with sqnr_eval.",
    )

    parser.add_argument(
        "--calibration_num_threads",
        type=int,
        default=0,
        help="Thread count for PTQ calibration. 0 = auto-tune (default).",
    )

    parser.add_argument(
        "--freeze_all_params",
        action="store_true",
        help="(QAT CI only) Freeze all model weight parameters during QAT training so that "
        "only quantization scale/zero_point are updated via observer calibration — weights are not "
        "tuned. This is used as a controlled baseline to isolate the effect of weight updates: "
        "comparing --freeze_all_params vs normal QAT shows how much benefit comes from weight "
        "adaptation alone.",
    )

    parser.add_argument(
        "--quant_recipe_suggestion",
        action="store_true",
        help="Enable automatic quant recipe suggestion in PTQ",
    )

    return parser


def export_llama(args) -> None:
    if is_qwen35_model(args.decoder_model):
        if args.backend != "htp" or args.qat or args.use_fp16 or args.enable_x86_64:
            raise ValueError("GDN VTCM requires quantized HTP device execution.")
        if args.model_mode not in {"hybrid", "kv"} or args.batch_size != 1:
            raise ValueError("GDN VTCM supports hybrid/kv with batch_size=1.")
        if args.max_seq_len < 128:
            raise ValueError("Qwen3.5 requires max_seq_len >= 128.")
        args.prefill_ar_len = 128
        args.shared_buffer = True
    if not set(args.calib_text_preserve_sources or []).issubset(
        args.calib_text_datasets or []
    ):
        raise ValueError("Preserved sources must also appear in --calib_text_datasets.")
    if args.skip_qdq_save and SQNR_EVAL in args.eval_methods:
        raise ValueError("--skip_qdq_save cannot be combined with sqnr_eval")
    if args.calib_text_datasets:
        if any(n <= 0 for n in args.calib_text_samples) or len(
            args.calib_text_samples
        ) not in (1, len(args.calib_text_datasets)):
            raise ValueError(
                "--calib_text_samples requires positive counts: one for all sources or one per dataset."
            )
    if args.calibration_num_threads < 0:
        raise ValueError("--calibration_num_threads must be >= 0")
    if args.calibration_device == "cuda" and not args.pre_gen_pte:
        if args.qat:
            raise ValueError("--calibration_device cuda supports PTQ, not QAT")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA calibration requested but CUDA is unavailable. Check the PyTorch CUDA build and CUDA_VISIBLE_DEVICES, or use --calibration_device cpu."
            )
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")
    if args.optrace and args.enable_x86_64:
        raise RuntimeError("--optrace is only supported for on-device execution.")
    if args.optrace and args.device is None:
        raise RuntimeError(
            "device serial is required for --optrace. Please specify -s/--device."
        )
    if (TASKS_EVAL or SQNR_EVAL) in args.eval_methods and args.model_mode not in {
        "kv",
        "hybrid",
    }:
        raise RuntimeError(
            "Eval device perplexity is only supported for KV mode. Hybrid mode will only use KV mode when evaluating tasks/sqnr."
        )
    if TASKS_EVAL in args.eval_methods and args.eval_tasks is None:
        if args.calib_tasks is None:
            logging.warning(
                "--eval_tasks is set but --calib_tasks is not; quantization "
                "calibration will use --prompt instead of a task dataset. "
                "Pass --calib_tasks to match the previous --tasks behavior."
            )
        raise RuntimeError("Please provide --eval_tasks to eval perplexity")
    if args.decoder_model == "gemma4-e2b" and not args.embedding_quantize:
        raise RuntimeError(
            "gemma4-e2b requires --embedding-quantize: "
            "per-layer embedding is too large for HTP client buffer."
        )
    assert (
        args.decoder_model in SUPPORTED_LLM_MODELS
    ), f"Unknown decoder_model: {args.decoder_model}."
    decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
    logging.info(f"*** {args.decoder_model} ***\n%s", str(decoder_model_config))

    if (
        args.decoder_model.startswith(("lincal3_5", "lincal3_vl"))
        and not args.enable_true_swa
    ):
        logging.info(
            "Enabling true SWA for %s because it uses sliding_attention layers.",
            args.decoder_model,
        )
        args.enable_true_swa = True

    if args.max_context_len is None:
        args.max_context_len = args.max_seq_len + (
            128 if is_qwen35_model(args.decoder_model) else 0
        )
    if is_qwen35_model(args.decoder_model) and args.model_mode == "hybrid":
        qwen35_prefill_capacity(args.max_seq_len, args.max_context_len)
    if args.use_attention_sink is None:
        assert (
            args.max_context_len >= args.max_seq_len
        ), "Please ensure max_context_len is >= max_seq_len"

    # Specify pte filenames
    if args.model_mode == "kv":
        pte_filename = "kv_llama_qnn"
    elif args.model_mode == "hybrid":
        assert (
            args.max_context_len >= args.prefill_ar_len
        ), "Please ensure max_context_len is >= prefill_ar_len"
        pte_filename = "hybrid_llama_qnn"
    elif args.model_mode == "lookahead":
        assert (
            args.max_context_len >= args.prefill_ar_len
        ), "Please ensure max_context_len is >= prefill_ar_len"
        assert args.max_context_len > next_power_of_two(
            (args.window + args.gcap) * (args.ngram - 1)
        ), "Please ensure max_context_len is > next_power_of_two((args.window + args.gcap) * (args.ngram - 1))"
        pte_filename = "lookahead_llama_qnn"
    else:
        raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")

    if args.model_mode == "hybrid" and args.online_prepare:
        raise RuntimeError(
            "Currently hybrid mode is not compatible with online_prepare."
        )

    if args.decoder_model == "stories260k":
        pte_filename = f"{args.decoder_model}_" + pte_filename
    pte_filenames = {
        TEXT_DECODER: pte_filename,
        AUDIO_ENCODER: f"{AUDIO_ENCODER}_qnn",
        TEXT_ENCODER: f"{TEXT_ENCODER}_qnn",
        VISION_ENCODER: f"{VISION_ENCODER}_qnn",
        TOK_EMBEDDING: f"{TOK_EMBEDDING}_qnn",
    }
    # Prepare tokenizer
    tokenizer_wrapper = TokenizerWrapper(
        args,
        decoder_model_config,
    )
    runtime_tokenizer_path = tokenizer_wrapper.runtime_tokenizer_path

    # Prepare dataset
    text_decoder_pte_path = f"{args.artifact}/{pte_filenames[TEXT_DECODER]}.pte"
    attention_sink_evictor_pte_path = f"{args.artifact}/{ATTENTION_SINK_EVICTOR}.pte"
    tok_embedding_pte_path = f"{args.artifact}/{pte_filenames[TOK_EMBEDDING]}.pte"
    encoder_pte_paths = {}
    for modality in [AUDIO_ENCODER, VISION_ENCODER]:
        if hasattr(decoder_model_config, modality):
            encoder_pte_paths[modality] = (
                f"{args.artifact}/{pte_filenames[modality]}.pte"
            )

    is_multimodal = any(
        [
            hasattr(decoder_model_config, VISION_ENCODER),
            hasattr(decoder_model_config, AUDIO_ENCODER),
        ]
    )
    if is_multimodal:
        # TODO: Implement attention sink support for multimodal models (vision/audio).
        if args.use_attention_sink is not None:
            raise ValueError(
                "Multimodal models currently do not support attention sink feature."
            )
        if args.eval_tasks is not None:
            raise ValueError("Multimodal models do not support --eval_tasks.")
        if args.qat:
            raise ValueError("QAT is not supported for multimodal models (VLM/ALM).")

    if not args.pre_gen_pte:
        if is_multimodal:
            if args.calib_samples is None:
                raise ValueError(
                    "For MLLMs calibration data is required for compilation. "
                    "Provide --calib_samples with a vision/audio JSON file."
                )
        else:
            has_calib = any(
                (
                    args.calib_tasks,
                    args.calib_samples,
                    args.calib_hf_dataset,
                    args.calib_text_datasets,
                )
            )

            if args.qat:
                has_train = args.train_tasks or args.train_hf_dataset
                if not has_train:
                    raise ValueError(
                        "QAT requires training data: provide --train_tasks / "
                        "--train_hf_dataset together with --calib_* for the "
                        "calibration data."
                    )

            # Calibration data is required for compilation.
            if not has_calib:
                raise ValueError(
                    "For LLMs calibration data is required for compilation. "
                    "Provide --calib_tasks, --calib_samples, --calib_hf_dataset or --calib_text_datasets."
                )

    if args.pre_gen_pte:
        text_decoder_pte_path = f"{args.pre_gen_pte}/{pte_filenames[TEXT_DECODER]}.pte"
        attention_sink_evictor_pte_path = (
            f"{args.pre_gen_pte}/{ATTENTION_SINK_EVICTOR}.pte"
        )
        tok_embedding_pte_path = (
            f"{args.pre_gen_pte}/{pte_filenames[TOK_EMBEDDING]}.pte"
        )
        encoder_pte_paths = {}
        for modality in [AUDIO_ENCODER, VISION_ENCODER]:
            if hasattr(decoder_model_config, modality):
                encoder_pte_paths[modality] = (
                    f"{args.pre_gen_pte}/{pte_filenames[modality]}.pte"
                )

        if args.use_attention_sink:
            compile_attention_sink_evictor(
                args,
                decoder_model_config,
                text_decoder_pte_path,
                attention_sink_evictor_pte_path,
            )
        inference(
            args,
            decoder_model_config,
            runtime_tokenizer_path,
            tokenizer_wrapper,
            text_decoder_pte_path,
            encoder_pte_paths,
            tok_embedding_pte_path,
            attention_sink_evictor_pte_path,
            is_multimodal,
        )
        print(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")
        return

    decode_qdq_path = os.path.join(args.artifact, DECODE_QDQ_FILENAME)
    decode_qdq_before = _file_state(decode_qdq_path)
    text_decoder_pte_before = _file_state(text_decoder_pte_path)
    compile(
        args,
        decoder_model_config,
        pte_filenames,
        tokenizer_wrapper,
        is_multimodal,
    )
    if args.use_attention_sink:
        compile_attention_sink_evictor(
            args,
            decoder_model_config,
            text_decoder_pte_path,
            attention_sink_evictor_pte_path,
        )

    decode_qdq_after = _file_state(decode_qdq_path)
    text_decoder_pte_after = _file_state(text_decoder_pte_path)
    if (
        text_decoder_pte_after is None
        or text_decoder_pte_after == text_decoder_pte_before
    ):
        raise RuntimeError(
            "Compilation did not produce a fresh text decoder PTE; refusing stale lineage"
        )
    decode_qdq_produced = (
        not args.skip_qdq_save
        and not args.use_fp16
        and decode_qdq_after is not None
        and decode_qdq_after != decode_qdq_before
    )
    if not decode_qdq_produced:
        logging.warning(
            "No fresh %s was produced by this export; lineage will be incomplete",
            DECODE_QDQ_FILENAME,
        )
    lineage_path = write_export_lineage(
        args,
        pte_path=text_decoder_pte_path,
        pt2_path=decode_qdq_path if decode_qdq_produced else None,
        runtime_tokenizer_path=runtime_tokenizer_path,
        qnn_sdk_build_id=describe_sdk_build_id(),
        repo=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),
    )
    logging.info("Export lineage saved to %s", lineage_path)

    if args.compile_only:
        if args.ip and args.port != -1:
            # Prepare validation results for CI system
            text_decoder_pte_path = f"{args.artifact}/{pte_filename}.pte"
            text_decoder_pte_path = os.path.getsize(text_decoder_pte_path)
            validation_results = {
                "pte_size": text_decoder_pte_path,
            }
            if is_multimodal:
                tok_embedding_pte_path = (
                    f"{args.artifact}/{pte_filenames[TOK_EMBEDDING]}.pte"
                )
                validation_results.update(
                    {
                        "tok_embedding_pte_size": os.path.getsize(
                            tok_embedding_pte_path
                        ),
                    }
                )
                for modality, encoder_pte_path in encoder_pte_paths.items():
                    validation_results.update(
                        {f"{modality}_pte_size": os.path.getsize(encoder_pte_path)}
                    )

            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps(validation_results))

        print(f"Finish compile_only and save to {args.artifact}")
        return

    inference(
        args,
        decoder_model_config,
        runtime_tokenizer_path,
        tokenizer_wrapper,
        text_decoder_pte_path,
        encoder_pte_paths,
        tok_embedding_pte_path,
        attention_sink_evictor_pte_path,
        is_multimodal,
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.build_folder = os.path.realpath(args.build_folder)
    try:
        export_llama(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)


# flake8: noqa: C901
if __name__ == "__main__":
    main()
