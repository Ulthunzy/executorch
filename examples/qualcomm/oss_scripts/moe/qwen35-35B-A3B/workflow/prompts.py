from workflow.vision import image_prompt, pad_prompt


def prepare_prompt(args, tokenizer, np, vision=None):
    context = args.model_settings.context
    if args.prefill_token_file:
        ids = np.fromfile(args.prefill_token_file, dtype="<i8").tolist()
        if (
            not ids
            or (args.prompt_tokens is not None and len(ids) != args.prompt_tokens)
            or not context.prefill_ar_len <= len(ids) <= context.max_prefill_tokens
            or len(ids) % context.prefill_ar_len
            or min(ids) < 0
            or max(ids) >= len(tokenizer)
        ):
            raise ValueError(
                "Token input must contain valid IDs and a complete AR128 sequence"
            )
        return [], ids
    if vision is not None:
        messages, ids, _ = image_prompt(
            tokenizer,
            args.prompt,
            vision["output_shape"][1],
            args.prompt_tokens,
            enable_thinking=args.enable_thinking,
            system_prompt=args.system_prompt,
        )
        return messages, ids
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    return messages, pad_prompt(
        tokenizer,
        ids,
        args.prompt_tokens,
        context.prefill_ar_len,
        context.max_prefill_tokens,
    )
