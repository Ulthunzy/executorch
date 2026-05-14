## Summary

This directory contains a code-first skeleton for adapting Qwen2.5-VL into the
`examples/models` tree.

The adapter mirrors the Hugging Face architecture instead of pretending the
model already fits the current ExecuTorch multimodal runner contract.

## What is implemented

- A thin `Qwen2_5VLAdapter` wrapper around `Qwen2_5_VLForConditionalGeneration`
- Explicit component wrappers for:
  - `vision_encoder(pixel_values, image_grid_thw)`
  - `video_encoder(pixel_values_videos, video_grid_thw)`
  - `token_embedding(input_ids)`
  - `text_decoder(inputs_embeds, attention_mask, position_ids, past_key_values, rope_deltas)`
- Helper methods to:
  - prepare processor inputs
  - build multimodal position inputs
  - inspect the prefill state used before autoregressive decode

## Cache flow

Qwen2.5-VL does **not** maintain a separate long-lived KV cache for the vision
tower.

The flow is:

1. `vision_encoder` runs once and produces visual embeddings.
2. The model splices those embeddings into the text sequence at vision
   placeholder positions.
3. `text_decoder` performs prefill on the combined sequence.
4. The decoder KV cache now contains K/V for:
   - text prefix tokens
   - visual prefix tokens
   - any text suffix tokens that appear after the image/video placeholders
5. Subsequent decode steps only run the text decoder with `past_key_values`.

## Current gap with the generic ExecuTorch multimodal runner

The current generic multimodal runner expects:

- `vision_encoder` to take a single image tensor
- `text_decoder` to take embeddings plus a simple decode position tensor

Qwen2.5-VL needs more than that:

- `image_grid_thw` or `video_grid_thw`
- multimodal position construction
- optional `rope_deltas`
- processor-driven vision placeholder layout

So this directory is the right architectural split, but it is **not yet**
directly wired into `extension/llm/runner/multimodal_prefiller.cpp`.

## Recommended next step

To make this runnable end-to-end inside ExecuTorch, extend the multimodal input
path so the runner can pass:

- image tensor + `image_grid_thw`
- video tensor + `video_grid_thw`
- multimodal position metadata required by the decoder
