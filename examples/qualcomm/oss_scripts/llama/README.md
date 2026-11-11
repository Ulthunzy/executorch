# Summary

## Model validation workflow

SafeTensors/PT2/PTE PPL、校准数据准备、多轮测试、生成策略消融和错误报告的统一入口见 [model_validation/README.md](model_validation/README.md)。新模型通过配置文件选择路径和参数；扩展任务见 [EXTENDING.md](model_validation/EXTENDING.md)。

## Overview

**Video Tutorial:** [Build Along: Run LLMs Locally on Qualcomm Hardware Using ExecuTorch](https://www.youtube.com/watch?v=41PKDlGM3oU)

This file provides you the instructions to run LLM Decoder model, VLM model, and ALM model with different parameters via Qualcomm HTP backend. We currently support the following models:
- Large language models
<!-- numbered list will be automatically generated -->
 1. LLAMA2 Stories 110M
 1. LLAMA3.2 1B
 1. LLAMA3.2 3B
 1. Codegen2 1B
 1. Gemma 2B
 1. Gemma2 2B
 1. Gemma3 1B
 1. Gemma4 E2B
 1. GLM 1.5B
 1. Granite3.3 2B
 1. Phi4-mini-instruct
 1. QWEN2.5 0.5B / 1.5B
 1. QWEN3 0.6B / 1.7B
 1. SmolLM2 135M
 1. SmolLM3 3B
- Vision-Language Models
<!-- numbered list will be automatically generated -->
 1. SmolVLM 500M
 1. InternVL3 1B
- Audio-Language models
<!-- numbered list will be automatically generated -->
 1. Granite-speech-3.3-2b

We offer the following modes to execute the model:

- KV Cache Mode: In KV Cache mode, the model takes in a single previous token and generates the next predicted token along with its KV cache. It is efficient for generating subsequent tokens after the initial prompt.

- Hybrid Mode: Hybrid mode leverages the strengths of both AR-N model and KV cache modes to optimize token generation speed. Initially, it uses AR-N model to efficiently generate the prompt's key-value (KV) cache. Then, the mode switches to KV cache mode, which excels at generating subsequent tokens.
  - AR-N model: The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use it to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor in hybrid mode.
  - Prompt processing with AR-N model:
  <figure>
    <img src="assets/PromptProcessingWithARN.png" alt="Prompt Processing With AR-N Model">
    <figcaption>Prompt processing is done using a for-loop. An N-token block is taken, and the KV cache is updated for that block. This process is repeated until all tokens are consumed, with the last block potentially requiring padding. For flexibility, the AR-N model can handle any input length less than the maximum sequence length. For TTFT, the input length (or number of blocks) will vary depending on the actual input length, rather than always being the same.
    </figcaption>
</figure>

- Lookahead Mode: Lookahead Mode introduces [lookahead decoding](https://arxiv.org/abs/2402.02057) and uses AR-N model to process prompt to enhance token generation speed. While decoding multiple tokens in a single step is infeasible, an LLM can generate multiple guess tokens in parallel. These guess tokens may fit into future parts of the generated sequence. The lookahead decoder generates and verifies these guess tokens, integrating them into the sequence if suitable. In some cases, it can obtain more than one token in a single step. Result is lossless.

### EOS stopping and fixed-length benchmarking

Generation stops at EOS by default. For fixed-length benchmarking, add
`--ignore_eos` to `llama.py`, or `--ignore_eos=true` to `qnn_llama_runner` /
`qnn_multimodal_runner`. This continues generation until the total sequence
length limit (prompt plus output), even after EOS. Use `--ignore_eos=false`
with the runners to explicitly restore normal stopping; the Python equivalent
is `--no-ignore_eos`.

The switch applies to KV, hybrid, and lookahead generation, including an EOS
sampled immediately after prefill. It only changes runtime behavior: the same
PTE can be reused without recalibration or recompilation. Text generated after
EOS in benchmark mode should not be used to assess response quality.

### Reusable self-identity evaluation

For a fixed 50-question, two-turn device benchmark with retained KV history,
raw outputs, reviewable accuracy reports, and cross-model summaries, see
[model_validation](model_validation/README.md). It supports ChatML QNN
Lincal/Qwen models and preserves the model's default thinking behavior.

## Hardware Support

We’ve validated this flow on the **Samsung Galaxy S23**, **Samsung Galaxy S24**, **Samsung Galaxy S25**, and **OnePlus 12**.
Support on other hardware depends on the **HTP architecture (HtpArch)** and the feature set available on that version.

### HTP Minimum Version Requirements

- **LPBQ (16a4w block-wise quantization)** requires **V69 or newer**
- **Weight sharing** between prefill and decode requires **V73 or newer**
- **16-bit activations + 16-bit weights for matmul** (e.g., 16-bit KV cache) requires **V73 or newer**

### Prefill/Decode Weight Sharing

`LLMModelConfig.use_weight_sharing` defaults to `True`, including for `lincal-1_7b`.
`llama.py` reads this model setting when building the HTP decoder and token
embedding compile specs. In hybrid mode, QNN can share eligible identical weights
between prefill and decode graphs; KV-only mode has no prefill graph to share with.
Actual savings depend on the compiled graphs and QNN SDK.

To disable sharing for a model, set `use_weight_sharing = False` in its registered
configuration class in `__init__.py`, then re-export the PTE. This is useful for
hardware compatibility or model-specific decode performance regressions, such as
the previously observed Qwen3.5 regression. This setting does not change QNN GPU
compile options or runtime generation parameters.

### Quantization Guidance for Older Devices

For older HTP versions, you may need to adjust the quantization strategy. Recommended starting points:

- Use **16a4w** as the baseline
- Optionally apply **SpinQuant**
- Use **16a8w selectively on some layers** to further improve accuracy (mixed-precision quantization)

### Memory Limit Errors (4 GB HTP Limit)

If you encounter errors like the following, it typically means the model’s requested memory exceeds the **4 GB per-context limit** on HTP.
To resolve this, try **increasing the sharding number** (`num_sharding`) to reduce per-shard memory usage:

```
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to find available PD for contextId 1 on deviceId 0 coreId 0 with context size estimate 4025634048
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> context create from binary failed on contextId 1
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Fail to create context from binary with err 1002
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Size Calculation encounter error! Doing Hard reset of reserved mem to 0.
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to create context from binary with err 0x3ea
[ERROR] [Qnn ExecuTorch]: Can't create context from binary
```


## Instructions
### Note
1. For hybrid mode, the export time will be longer and can take up to 1-4 hours to complete, depending on the specific model users are exporting.
2. When exporting a hybrid mode model, memory consumption will be higher. Taking LLAMA3.2 1B as an example, please ensure the device has at least 80 GB of memory and swap space.


### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/main/backends-qualcomm) to build Qualcomm AI Engine Direct Backend.
3. Please install the llm eval dependency via [examples/models/llama/install_requirements.sh](https://github.com/pytorch/executorch/blob/main/examples/models/llama/install_requirements.sh)

### Step 2: Prepare Model

#### LLAMA2
Download and prepare stories110M model

```bash
# tokenizer.model & stories110M.pt:
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"

# tokenizer.bin:
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

# params.json:
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
```

#### LLAMA3.2
Follow the [instructions](https://www.llama.com/) to download models.
At the end of this step, users should have the following files ready: `consolidated.00.pth`, `params.json`, and `tokenizer.model`.


### Step3: Run default examples.
#### Note:
All example scripts below use hybrid mode, which is optimized for on-device performance. However, compiling a model in hybrid mode can consume a significant amount of memory on the host machine—sometimes up to ~100 GB. If your host machine has limited memory, it is highly recommended to switch from `--model_mode hybrid` to `--model_mode kv` and remove the `--prefill_ar_len` flag.

#### LLAMA2
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint stories110M.pt --params params.json --tokenizer_model tokenizer.model --tokenizer_bin tokenizer.bin --decoder_model stories110m --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "Once upon a time" --calib_tasks wikitext --calib_limit 1
```

#### LLAMA3.2 1B Instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### LLAMA3.2 3B Instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```


#### Codegen2
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model codegen2_1b --model_mode kv --max_seq_len 1024 --prompt "def hello_world():" --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/text.json
```

#### Gemma 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma-2b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### Gemma2 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma2-2b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### Gemma3 1B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma3-1b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### Gemma4 E2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma4-e2b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1 --embedding-quantize '4,32'
```

#### GLM 1.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model glm-1_5b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### Granite3.3 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model granite_3_3-2b_instruct --prompt "I would like to learn python, could you teach me with a simple example?" --eval_methods tasks_eval --eval_tasks hellaswag --eval_limit 10 --calib_tasks hellaswag --calib_limit 10
```

#### Phi4-mini-instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model phi_4_mini --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### QWEN2.5 0.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model qwen2_5-0_5b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### QWEN2.5 1.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen2_5-1_5b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### QWEN3 0.6B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model qwen3-0_6b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### QWEN3 1.7B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --temperature 0 --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen3-1_7b --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### SmolLM2
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model smollm2_135m --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### SmolLM3
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model smollm3-3b --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1
```

#### Using custom calibration samples for LLMs

Instead of `--calib_tasks`, you can supply your own conversation JSON files via `--calib_samples`. The samples are fed into the quantization calibration pass to collect activation observer statistics — they do not affect the inference prompt. This is useful when you want to calibrate on domain-specific or instruct-format data rather than a generic lm_eval task.

```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model smollm2_135m --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/text.json
```

You can also provide both `--calib_tasks` and `--calib_samples` at the same time; the pipeline concatenates both data sources for calibration.


## Multimodal Support

### Overview

Multimodal models extend LLM by processing multiple input modalities (vision, audio, text) simultaneously. This framework provides a unified architecture for multimodal via Qualcomm HTP backend.

**Current Support Status:**
- **Vision-Language Models (VLM)**: Fully supported
- **Audio-Language Models (ALM)**: Fully supported

### Multimodal Architecture

For general multimodal processing pipeline please refer [Multimodal Architecture](../../../../extension/llm/runner/README.md#multimodalrunner-architecture)


### Processing Pipeline

Multimodal inference follows these key stages:

1. **Modality-Specific Encoding**
   - **Vision**: Images are processed through a vision encoder to generate visual embeddings
   - **Audio**: Audio waveforms are processed through an audio encoder to generate audio embeddings
   - **Text**: Text prompts are tokenized and embedded

2. **Embedding Fusion**
   - All modality embeddings are projected to a common embedding dimension
   - Embeddings are concatenated or fused according to the model's template
   - Special tokens are inserted to mark modality boundaries

3. **Unified Language Generation**
   - The fused embeddings are fed into the language model decoder
   - The decoder generates text autoregressively using the same execution modes as LLM models (KV Cache, Hybrid, Lookahead)

---

## Audio-Language Model (ALM) Support

Audio-Language Models (ALMs) combine speech/audio processing and natural language processing to understand and generate text based on audio inputs. ALMs in this framework consist of:

### Dependencies

ALM models require the `soundfile` package for audio loading:

```bash
pip install soundfile
```


- **[Audio Encoder](model/audio_encoder.py)**: Processes raw audio waveforms into audio embeddings (e.g., CTC encoder for Granite-speech)
  - **Projector** (included in audio encoder): Aligns audio embeddings with the language model's embedding space.
- **[Language Decoder](model/static_llama.py)**: Reuse static llama to generates text based on fused visual and text embeddings.

### Instructions

#### Granite-speech-3.3-2b

Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model granite_speech_3_3-2b --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "can you transcribe the speech into a written format?" --audio_path "https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/10226_10111_000000.wav?download=true" --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/audio.json
```

### Specifying Custom Audio

You can specify a custom audio file for ALM models using the `--audio_path` flag:
- **HTTP/HTTPS URLs**: Direct links to audio on the web
  - Example: `"https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/10226_10111_000000.wav?download=true"`
- **HuggingFace repository filenames**: Files that exist in the model's HuggingFace repository are automatically downloaded
  - Example: `"10226_10111_000000.wav"` (auto-downloaded from `ibm-granite/granite-speech-3.3-2b`)
- **Local file paths**: Absolute or relative paths to `.wav` files on your system
  - Example: `"/path/to/your/audio.wav"`

#### Audio Preprocessing

The audio encoder configuration is defined in `encoder/encoder_config.py`:

```python
# In encoder/encoder_config.py
@dataclass(init=False, frozen=True)
class GraniteSpeechEncoder(AudioModalityConfig):
    encoder_class = GraniteSpeechCTCEncoderWrapper
    audio_seq_len = 171
    quant_recipe = GraniteSpeechEncoderQuantRecipe
```

- **audio_seq_len**: Number of audio tokens generated by the encoder.

The audio is automatically:
1. Loaded from the specified file path or downloaded from HuggingFace
2. Read as a waveform using `soundfile` and converted to a float tensor of shape `[1, T]`
3. Processed by the HuggingFace `AutoProcessor` to produce mel-filterbank features of shape `(1, 844, 160)`
4. Passed through the CTC encoder and QFormer projector to produce audio embeddings of shape `[1, audio_seq_len, hidden_dim]`

### Using Pre-Generated PTE Files

If you have already compiled a ALM model, you can run inference with pre-generated PTE files:

```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model granite_speech_3_3-2b --model_mode hybrid  --prefill_ar_len 128 --max_seq_len 1024 --prompt "can you transcribe the speech into a written format?" --audio_path "https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/10226_10111_000000.wav?download=true" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

### ALM Processing Details

The ALM inference pipeline consists of:

1. **Audio Encoding Phase**
   - Input audio waveform is loaded and preprocessed into mel-filterbank features: `(1, 844, 160)`
   - CTC encoder extracts acoustic features using Conformer blocks with block-wise local attention
   - QFormer projector aligns audio embeddings to the language model dimension: `[batch, audio_seq_len, hidden_dim]`

2. **Text Tokenization Phase**
   - User prompt is tokenized into text tokens
   - Text tokens are embedded: `[batch, text_seq_len, hidden_dim]`

3. **Embedding Fusion Phase**
   - Audio and text embeddings are concatenated according to the model's template
   - The `<audio>` placeholder in the prompt is expanded to the model-specific special token `<|audio|>`
   - Final fused sequence: `[batch, audio_seq_len + text_seq_len, hidden_dim]`

4. **Language Generation Phase**
   - Fused embeddings are fed into the language decoder
   - Autoregressive generation produces output tokens using KV cache mode
   - KV cache is updated for efficient subsequent token generation


## Vision-Language Model (VLM) Support

Vision-Language Models (VLMs) combine computer vision and natural language processing to understand and generate text based on visual inputs. VLMs in this framework consist of:

- **[Vision Encoder](model/vision_encoder.py)**: Processes images into visual embeddings (e.g., SigLIP for SmolVLM)
  - **Projector** (included in vision encoder): Aligns visual embeddings with the language model's embedding space
- **[Language Decoder](model/static_llama.py)**: Reuse static llama to generates text based on fused visual and text embeddings

### Instructions

#### SmolVLM 500M
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model smolvlm_500m_instruct --model_mode hybrid --prefill_ar_len 16 --max_seq_len 1024 --prompt "Can you describe this image?" --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg" --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json
```

#### InternVL 1B
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model internvl3_1b --model_mode hybrid --prefill_ar_len 32 --max_seq_len 1024 --prompt "Can you describe this image?" --image_path "http://images.cocodataset.org/val2017/000000039769.jpg" --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json
```

### Specifying Custom Image

You can specify custom image for VLM models using the `--image_path` flag:

Take a example image of Statue-of-Liberty in New York Bay
- **HTTP/HTTPS URLs**: Direct links to images on the web
  - Example: `https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg`
- **Local file paths**: Absolute or relative paths to image files on your system
  - Example: [`./examples/qualcomm/oss_scripts/llama/assets/samples/images/Statue-of-Liberty-Island-New-York-Bay.png`](assets/samples/images/Statue-of-Liberty-Island-New-York-Bay.png)

#### Image Preprocessing

Each VLM model has specific preprocessing requirements defined in its configuration:

```python
# In encoder/encoder_config.py
@dataclass(init=False, frozen=True)
class SmolVLMEncoder(VisionModalityConfig):
    encoder_class = Idefics3VisionEncoder
    img_seq_len = 64
    img_resized_h = 512
    img_resized_w = 512
    quant_recipe = SmolVLMEncoderQuantRecipe
```

- **img_resized_h / img_resized_w**: Target resolution for the vision encoder
- **img_seq_len**: Number of visual tokens generated by the encoder

The image is automatically:
1. Loaded from the specified URL or file path
2. Resized to the model's expected resolution and preprocessed by HuggingFace [processors](https://huggingface.co/docs/transformers/main/processors)

### Using Pre-Generated PTE Files

If you have already compiled a VLM model, you can run inference with pre-generated PTE files:

```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model smolvlm_500m_instruct --model_mode kv --max_seq_len 1024 --prompt "Can you describe this image?" --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

### Multi-Turn Conversation with VLM

The framework supports multi-turn conversations with VLMs, allowing you to conduct dialogues that can involve multiple images.

- **Multi-Turn Prompts**: To engage in a conversation, provide multiple prompts sequentially using the `--prompt` argument. Each string will be treated as a separate turn.
- **Multiple Images**: You can supply multiple images (from URLs or local paths) using the `--image_path` argument.
- **Flexible Image Placement**: Use the `<image>` token within your prompt to specify exactly where each image's embeddings should be placed. The images provided via `--image_path` will replace the `<image>` tokens in the order they appear.

**Example**:

In this example, the first turn compares two images, the second turn asks a follow-up question about the first image, and the third turn asks for a caption for a third image.

```bash
# Define image URLs and prompts for a 3-turn conversation
IMAGE1_URL="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
IMAGE2_URL="http://images.cocodataset.org/val2017/000000039769.jpg"
IMAGE3_URL="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"

PROMPT1="<image><image>Compare these images above and list the differences."
PROMPT2="Answer the question: What's the main object in first image?"
PROMPT3="<image>Caption this image."

# Execute the multi-turn conversation
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --decoder_model smolvlm_500m_instruct --model_mode kv --max_seq_len 2048 --prompt "$PROMPT1" "$PROMPT2" "$PROMPT3" --image_path "$IMAGE1_URL" "$IMAGE2_URL" "$IMAGE3_URL" --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json
```

**How it works:**
- **Turn 1**: The prompt `"<image><image>Compare these images above and list the differences."` uses the first two images (`$IMAGE1_URL`, `$IMAGE2_URL`).
- **Turn 2**: The prompt `"Answer the question: What's the main object in first image?"` is a text-only follow-up. The conversation context is maintained from the previous turn.
- **Turn 3**: The prompt `"<image>Caption this image."` uses the third image (`$IMAGE3_URL`).

### VLM Processing Details

The VLM inference pipeline consists of:

1. **Vision Encoding Phase**
   - Input image is preprocessed (resize, normalize)
   - Vision encoder generates visual embeddings: `[batch, img_seq_len, hidden_dim]`
   - Visual embeddings are projected to match the language model dimension by the modality projector

2. **Text Tokenization Phase**
   - User prompt is tokenized into text tokens
   - Text tokens are embedded: `[batch, text_seq_len, hidden_dim]`

3. **Embedding Fusion Phase**
   - Visual and text embeddings are concatenated according to the model's template
   - Special tokens (e.g., `<image>`, `<|fake_token_around_image|>`, `<fake_token_around_image>`) mark modality boundaries (see [tokenizer.py](tokenizer.py))

   ```python
   # Special tokens for Vision-Language Model
   VLM_SPECIAL_TOKENS = {
       "smolvlm_500m_instruct": {
           "image_token": "<image>",
           "global_img": "<global-img>",
           "fake_wrap_start": "<fake_token_around_image>",
           "fake_wrap_end": "<fake_token_around_image>",
       },
       ...
   }
   ```
   - Final fused sequence: `[batch, img_seq_len + text_seq_len, hidden_dim]`

4. **Language Generation Phase**
   - Fused embeddings are fed into the language decoder
   - Autoregressive generation produces output tokens
   - KV cache is updated for efficient subsequent token generation


## KV Cache update mechanism
We use Smart Mask mechanisms for updating the key-value (KV) cache.

#### Smart Mask mechanism:
<figure>
    <img src="assets/SmartMask.png" alt="Smart Mask mechanism">
    <figcaption>The figure illustrates how key and value caches are updated during each inference step. The Smart Mask mechanism simplifies updating tokens in the cache by modifying only the new token at the designated position. This approach is useful for shared buffers, though it does require copying data in CPU memory to update the kv cache. </figcaption>
</figure>

#### Analysis KV Cache Update Mechanism for each Layer each inference
<table>
  <tr>
    <th>Mechanism</th>
    <th colspan="2" style="text-align:center;">Time Complexity</th>
    <th colspan="2" style="text-align:center;">Space Complexity</th>
  </tr>
  <tr>
    <th></th>
    <th style="text-align:center;">K</th>
    <th style="text-align:center;">V</th>
    <th style="text-align:center;">K</th>
    <th style="text-align:center;">V</th>
  </tr>
  <tr>
    <td style="text-align:center;">Smart Mask</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">num_head * seq_len * head_dim</td>
    <td style="text-align:center;">num_head * seq_len * head_dim</td>
  </tr>
</table>

### Additional Configs when running the script

#### Compile Only
If you would like to compile the model only, we have provided the flag `--compile_only`. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "I would like to learn python, could you teach me with a simple example?" --compile_only
```

#### Pre Generated PTE
On the other hand, if you already have a pre-compiled .pte model, you can perform inference by providing the flag `--pre_gen_pte` and specifying the folder that contains the .pte model. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "I would like to learn python, could you teach me with a simple example?" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

#### Lookahead Decoding Mode

You can choose the lookahead mode to enhance decoding speed. To use this mode, you need to specify the following parameters:
- `--ngram` (N-gram size): Represents the size of the n-grams used in the lookahead process.
- `--window` (window size): Determines how many future tokens the algorithm attempts to predict in each step.
- `--gcap` (Verification candidates): Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.

For more details, please refer to the paper ["Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057)

```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode lookahead --prefill_ar_len 32 --max_seq_len 128 --prompt "I would like to learn python, could you teach me with a simple example?" --ngram 3 --window 2 --gcap 2
```

#### Tasks Evaluation
This script supports task evaluation and is capable of assessing evaluation scores across 3 phases: prepare_pt2e(CPU FP), convert_pt2e(CPU QDQ), QNN on device.

To evaluate the perplexity across all 3 phases, users should provide the `--eval_methods tasks_eval` flag and specify the evaluation task. Please notice when this flag is provided, the `--prompt ${PROMPT}` will be ignored.

For example, using the Qwen model and 1 wikitext sample as the evaluation task, users can assess all 3 phases perplexity score in a single run by including the appropriate configuration:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods tasks_eval --calib_tasks wikitext --calib_limit 1 --eval_tasks wikitext --eval_limit 1 --verbose
```

From the example script above, 1 wikitext sample is used to evaluate all 3 phases. However, there are cases where a user may want to use one sample for quantization calibration and multiple samples for perplexity evaluation. In this case, the process should be split into two runs. In the 1st run, the model is compiled using one sample. In the 2nd run, the user can provide a different configuration for QNN device execution.
Example:
```bash
# 1st run to compile with --calib_limit 1
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods tasks_eval --calib_tasks wikitext --calib_limit 1 --compile_only
```
```bash
# 2nd run to perform QNN device execution with --eval_limit 3
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods tasks_eval --eval_tasks wikitext --eval_limit 3 --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```


#### SQNR Evalution
To evaluate QNN's output logits against the golden logits from `nn.Module`, users can provide the flag `--sqnr_eval`. Please note that SQNR evaluation will only compare the logits of the user's prompt and will not compare the new tokens generated by the model.
Example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods sqnr_eval
```



#### Quantization

The calibration data is independent from the runtime evaluation set, and only affects quantization quality, not the inference output.

Calibration data is required for compilation. There are two ways to supply it:

1. **`--calib_tasks`** — calibrate on one or more lm_eval tasks (tune with `--calib_limit` and `--calib_num_fewshot`). LLM-only.
2. **`--calib_samples`** — calibrate on custom conversation samples provided as JSON files (see format below). Required for multimodal models (VLM/ALM).

For LLMs, provide at least one of the two; for multimodal models, `--calib_samples` is mandatory.

| Purpose | Flags |
|---|---|
| Calibration data (lm_eval tasks) | `--calib_tasks`, `--calib_limit`, `--calib_num_fewshot` |
| Calibration data (custom samples) | `--calib_samples` (JSON files, HuggingFace message format) |

##### CUDA PTQ calibration

Add `--calibration_device cuda` to run decoder PTQ calibration on an NVIDIA GPU.
For example, prefix the export command with `CUDA_VISIBLE_DEVICES=0` to select the
first GPU. The default is `--calibration_device cpu`; `--calibration_num_threads`
and CPU thread auto-tuning apply only to CPU calibration.

CUDA calibration preserves the model dtype and disables TF32 during calibration.
Model parameters, observer statistics and decoder inputs return to CPU before
quantization conversion and QNN compilation. This does not change the deployment
backend, prefill AR or SWA layout. `--backend gpu` selects a QNN deployment backend,
not the calibration device. CUDA calibration is PTQ-only, not QAT. Multimodal
encoder preprocessing runs on CPU; encoder calibration follows
`--calibration_device` and passes its quantized features to decoder calibration.

Calibration needs floating-point weights and activations, not just the final
4-bit model storage. If GPU memory is insufficient, reduce `--batch_size` or
select CPU calibration; the exporter does not silently change sequence lengths.

For device-only validation, add `--skip_qdq_save` to omit the full-sequence
`decode_qdq.pt2` evaluation artifact. Serializing this graph can consume substantial
host memory even with `--model_mode kv`. This option does not skip calibration or
change the deployed PTE; it cannot be combined with `--eval_methods sqnr_eval`.
The default still saves the QDQ artifact for later accuracy analysis.

##### Qwen3.5 and Lincal3.5 optimized export

`qwen3_5-0_8b`, `qwen3_5-2b`, `qwen3_5-4b`, `lincal3_5-0_8b` and `lincal3_5-2b`
share the optimized HTP export path through `llama.py`, with precision selected
by their quantization recipes:

- Merged K/z W8 per channel; the other large Linear weights use W4 group32.
  GDN out projections use fixed-grid GPTQ on calibration inputs.
- W8A16 visual encoder with the checkpoint's tanh GELU formula and INT32 bias
  range protection. Text and image samples retain their pairing; decoder
  calibration uses quantized visual features and covers AR128 and warm AR1.
- Native RMSNorm for decoder, Q/K and gated GDN normalization. External
  recurrent and convolution states use calibrated UINT16 encodings; attention
  cache precision is selected by the recipe.
- Prefill AR128 with FP16 GDN, valid-token state updates and one
  selected LM-head row. Hybrid decoder PTEs contain only `kv_forward` and
  `prefill_forward`; all prompts use the requested full-context prefill graph.
  No additional short-context prefill graphs are exported or selected at runtime.
  Decode uses `DeltaNetCoreGatedVtcm` on SM8850, `DeltaNetCoreVtcm` on other
  supported SoCs, and direct quantized greedy argmax.
  SM8850 with K=V=128 uses one fused `Chunk` call per GDN layer; other targets
  and dimensions retain the portable chunk64/leaf8 implementation.
- Lincal3.5 retains true SWA and its separate window mask. Calibration masks
  limit each token to the sliding window, including within an AR128 block.
  SWA caches retain their fixed length when deriving AR1. Pure SWA models
  skip internal full-attention trimming; mixed full/SWA
  models trim only full attention. MHA-to-SHA conversion is disabled for both
  families so they use the same grouped-attention transformation.

Current precision defaults are defined in `static_llm_quant_recipe.py`:

| Recipe / models | Input embedding weights | Attention K/V cache | Attention MatMul RHS | GDN conv/recurrent state |
|---|---|---|---|---|
| `Qwen3_5_0_8BQuantRecipe`: Qwen/Lincal 0.8B | Packed INT4, group32, HQQ | UINT8 | UINT8 | UINT16 |
| `Qwen3_5_2BQuantRecipe`: Qwen 2B | Packed INT4, group32, HQQ | UINT8 | UINT8 | UINT16 |
| `Qwen3_5QuantRecipe`: Qwen 4B, Lincal 2B | FP16 | UINT16 | INT16 | UINT16 |

The 0.8B and 2B recipes configure the embedding node directly:

```python
self.recipe.add_node_target(
    {torch.ops.aten.embedding.default},
    QuantDtype.use_16a4w_block,
    False,
    act_observer=MinMaxObserver,
    granularity=QuantGranularity.PER_BLOCK,
    extra_kwargs={"block_size": (1, 32)},
)
self.recipe.custom_quant_annotations.append(annotate_kv_8bit)
```

The separate embedding graph resolves the same node rules and priorities as
the decoder recipe. `add_regex_node_target` can override this rule by module
path. Weight bitwidth and group size come from the matched configuration;
there is no separate embedding setting in the recipe. An explicit legacy
`--embedding-quantize` remains an override. INT4 lookup uses the CPU portable
kernel with FP32 output, is applied during decoder calibration, and stays
packed in the embedding PTE; the tied LM head retains its original source
weights. Packed embedding inputs use INT64 in both KV and hybrid methods.
Calibration datasets and SQNR evaluation use the resolved token input dtype
even when the legacy CLI option is absent.

The 0.8B and 2B recipes use `annotate_kv_8bit`: both FA cache storage and
the attention MatMul RHS are UINT8. Query and softmax-probability inputs and
MatMul outputs remain UINT16. New K/V enter the calibrated8bit grid, historical
entries stay8bit through concatenation, and returned cache entries use the same
grid. Scale comes from calibration; the symmetric UINT8 grid uses zero point128.
Grouped attention keeps the calibrated RHS bit width. GDN conv/recurrent state
remain UINT16. There is no whole-history UINT8→16bit expansion path.
This applies when re-exporting KV and hybrid 0.8B/2B models, including 4K.
Previously exported PTEs retain their stored precision until re-exported.

The storage-only `annotate_kv_8bit_io` implementation and its expansion branches
were removed after the4K regression below. Its measurements describe historical
PTEs, not current defaults. Switching to native8bit K/V compute passed eight
recipe/embedding regressions covering AR128/AR1, prefill trimming and GDN
state precision. The 2B default was subsequently changed to INT4/KV8 after the
context performance matrix; 4B and Lincal 2B retain their previous defaults.
Full 0.8B 4K evaluation is recorded under
`llama-output/qwen3_5-0_8b-4k-kv8-compute-20261016/`.

##### Standard Qwen3.5 export entry point (2026-10-17)

The measured 0.8B/2B configuration now runs directly through `llama.py`.
No registry override or experimental export wrapper is required. Both models
use AR128 and export only `kv_forward` and `prefill_forward` in hybrid mode.
K/z and depthwise convolution stay W8, other Linear weights stay W4/group32,
and the existing out_proj GPTQ and image/text calibration remain enabled.

```bash
conda activate et_npu
source env.sh
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH"

python examples/qualcomm/oss_scripts/llama/llama.py \
  -a ./llama-output/qwen3_5-0_8b-m4096-hybrid-ar128 \
  --build_folder build-android \
  --soc_model SM8750 \
  --model_mode hybrid \
  --max_seq_len 4096 \
  --prefill_ar_len 128 \
  --decoder_model qwen3_5-0_8b \
  --calibration_device cuda \
  --calibration_num_threads 4 \
  --seed 42 \
  --prompt "Describe this image." \
  --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json \
  --calib_text_datasets wikitext-2 \
  --calib_text_samples 8 \
  --source_safetensors /hunter/model/Qwen/Qwen3.5-0.8B \
  --skip_qdq_save \
  --compile_only \
  > run-qwen3_5-0_8b.log 2>&1

python examples/qualcomm/oss_scripts/llama/llama.py \
  -a ./llama-output/qwen3_5-2b-m4096-hybrid-ar128 \
  --build_folder build-android \
  --soc_model SM8750 \
  --model_mode hybrid \
  --max_seq_len 4096 \
  --prefill_ar_len 128 \
  --decoder_model qwen3_5-2b \
  --calibration_device cuda \
  --calibration_num_threads 4 \
  --seed 42 \
  --prompt "Describe this image." \
  --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json \
  --calib_text_datasets wikitext-2 \
  --calib_text_samples 8 \
  --source_safetensors /hunter/model/Qwen/Qwen3.5-2B \
  --skip_qdq_save \
  --compile_only \
  > run-qwen3_5-2b.log 2>&1
```

Run exports serially. Change `--max_seq_len` and the artifact directory for
1024/2048/3072/8192; the default position-table capacity is logical length +128.
`--source_safetensors` supplies the local model and tokenizer directory.

On SM8850/v81, Qwen3.5 automatically replaces full-attention cores in both
methods with `attention_support`'s `AttentionValid` online-softmax operator.
Full-attention K/V precision comes exclusively from the quantization recipe:
`annotate_kv_8bit` selects UINT8 for both new K/V and historical caches.
The 0.8B and 2B recipes already include this annotation. Q/output and GDN
states remain UINT16; the export does not override the recipe's annotations.
Calibration still uses tensor caches; the exported methods receive FD
descriptors and `[valid, first, history]` lengths. `qwen35_export.json` records
each layer's cache capacity and encodings. The runner must be rebuilt from
this checkout to allocate mapped caches, append valid tokens, and invalidate
packed caches on reset. Cache allocations are uncached on the CPU and explicitly
mapped into the current CDSP session before publishing their FDs. On V81, the
2B's 8/2 Q/KV heads use the HMX kernel. Other unsupported HMX geometries retain
the generic FP32 path.
Re-export the PTE with the updated package when enabling HMX group4; scratch
workspace sizes are determined during compilation.
Other SoCs, attention sink and graphs containing SWA retain SDK attention
automatically.

Mapped KV capacity is the logical `--max_seq_len`. The separate
`get_prefill_position_capacity` metadata records the compiled position-table
capacity so a padded AR128 tail can extend beyond the logical token limit.
The runner checks valid tokens against KV capacity and padded positions against
the position-table capacity. New PTEs and runners must be used together; missing
position-capacity metadata is an error. The 0.8B/8K output-projection splitting
also uses the mapped-cache metadata and remains enabled with online attention.

On SM8850/v81, AR128 with K=V=128 uses the fused FP16 `Chunk` operator for
GDN preparation, triangular correction, readout and state update. Its
implementation is in `custom_op/deltanet_op_package_htp/DeltaNetOpPackage/src/kernels/Chunk.h`.
One graph call handles all 128 tokens; internal triangular blocks remain.
Rebuild the shared package and re-export the PTE to use `Chunk`.

For other shapes using the portable chunk path, SM8850/v81 uses a fused FP16 state
update during prefill. It packs weighted correction, runs HMX matmul and
applies the decay/add epilogue in one custom operator, retaining the original
FP16 rounding steps. This also applies when attention uses SDK operators.
Using these defaults requires rebuilding the shared operator package and
exporting a new PTE; swapping the library alone does not replace the SDK graph.

On SM8850/v81, decode automatically uses `DeltaNetCoreGatedVtcm` to fuse the
calibrated decay computation into the recurrence. Export generates exact SDK
AR1 tables under `<artifact>/decode_gate_tables` and caches them by target,
SDK libraries, gate arithmetic, constants and quantization grids. A cache miss
requires the target device through `--device`/`--host` and `ANDROID_NDK_ROOT`;
a valid cache can be reused without the device. This is export-time generation,
not inference warmup. The 2B model adds 36 MiB of table constants across its
18 GDN layers. Other supported SoCs use `DeltaNetCoreVtcm` with SDK gates.
No attention-core, KV-bitwidth, state-update or LUT-directory selector is needed.
The separate `oss_scripts/moe` export and runner have their own configuration.

For example, the 2B command above exports an 8K hybrid model by setting
`--max_seq_len 8192` and choosing a new artifact directory. Set
`--soc_model SM8850` to use the defaults described above.

Qwen3.5-family compilation also uses `compact_fx_provenance()` to stop PyTorch
from recursively copying FX node origin histories. It retains direct origins
and numerical/quantization metadata, while discarding nested debug ancestry
for newly created origins. The change is scoped to the compiling thread and
restored on normal exit or exceptions. Other model families and `--pre_gen_pte`
inference retain normal FX behavior. This is the same memory workaround used
in the matrix exports, where the largest 2B process RSS was 34.73 GiB; that
measurement is not a guaranteed memory ceiling for every host or SDK version.

Disposable experiment scripts have been removed after verification. PTEs,
measurements and calibration records are retained. See
[integration review and test records](README_QWEN35_TEST_RECORDS.md#qwen35-cli-integration-20261017).

##### GDN prefill implementation (2026-10-17)

Full-attention K/V retains the existing KV8 annotation; GDN recurrent and conv
state remain UINT16, and chunk MatMul remains FP16. INT4 embedding is unchanged.
The GDN state8, normalized-byte cache IO and integer MatMul experiments have
been removed from the implementation; their measurements remain historical
[test records](README_QWEN35_TEST_RECORDS.md#qwen35-gdn-experiment-cleanup-20261017).

The common AR128 prefill optimization is retained: Gather nodes become static
slices, redundant mask/sign operations are folded, and stable 8×8 inverse
leaves are embedded in 32×32 operands. Dense32 high-order powers are not used.
The UINT16 standalone GDN measured 0.892334 ms on SM8750; this is not full-model
throughput. Existing PTEs require re-export to use the updated chunk graph.

Historical storage-only UINT8 verification (2026-10-16): eight recipe/embedding regressions
passed. Actual Qwen full-attention AR128 and AR1 small PTEs ran on SM8750 with
one QNN delegate and no CPU fallback. Against PyTorch Q/DQ, the small model's
output SQNR was 75.92/76.00 dB; returned UINT8 K/V differed by at most one code
for AR128 and were identical for AR1. This is a conversion correctness check,
not SQNR against the original HF model. QNN optrace/QHAS confirmed both cache
load conversions and both output conversions used VTCM with zero DDR reads or
writes attributed to those conversion nodes. Capacity was 256, so this does
not establish 4K residency or a full-model speedup. A GDN + full-attention small
PTE also compiled as one delegate, with GDN UINT16 and attention UINT8 IO.
Local evidence: `llama-output/qwen3_5-0_8b-4k-embedding-kv-ablation-20261016/storage_io_smoke/README.md`.

Recipe integration checks (2026-10-16): seven embedding/cache tests passed,
alongside fourteen calibration-memory, SWA and native cache-manager checks.
The embedding/cache checks cover
AR128/AR1 with one GDN and one full-attention layer, preservation of GDN UINT16
grids, unchanged 2B/4B defaults, embedding node-rule/override resolution and packed
embedding PTE serialization. A small AR128 QNN PTE also compiled successfully
and its serialized FA cache inputs/outputs were `BYTE`, with GDN cache IO
`UINT16`; no post-export dtype repair was needed. These checks do not constitute
a new full-model 4K PPL or performance measurement. Local smoke artifacts:
`llama-output/qwen35-recipe-integration-20261016`.
The native cache test retained address/undefined-behavior sanitizers with
`ASAN_OPTIONS=detect_leaks=0`; LeakSanitizer cannot run under the sandbox's ptrace.

The experimental split/projection environment variables and GDN dtype/kernel
CLI switches have been removed. `--gdn_op_package_dir` selects the op package
source directory; its host, Android and DSP libraries are built incrementally
using the configured QNN, Hexagon SDK and Android NDK. The default integer
reduction preserves all UINT16 bits with radix-16 splitting. The ordinary entry
pins this reduction for host and DSP builds. Build configuration stamps track
compiler versions, SDK paths, source lists and effective flags, so changing
an experimental build configuration invalidates the previous objects.
SM8750 is the validated target for Qwen3.5. Both families require quantized HTP
device execution, batch size 1, and `hybrid` or `kv` mode. Prefill AR is fixed
at 128 and shared buffers are
enabled automatically. Default physical capacity
is `max_seq_len + 128`. Hybrid mode rounds internal FA capacity up to an AR128
boundary, with a minimum of 256; an explicit `max_context_len` must accommodate
that capacity. Calibration uses a single AR128 model; AR1 and the short method
share its weights and learned encodings. KV mode retains AR128 calibration but
skips generation prefill graphs, short buckets and the AR128 embedding export.

For a local Hugging Face model directory:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/qualcomm/oss_scripts/llama/llama.py \
  -a ./llama-output/qwen35-optimized \
  --build_folder build-android --soc_model SM8750 \
  --decoder_model qwen3_5-0_8b \
  --source_safetensors /path/to/Qwen3.5-0.8B \
  --model_mode hybrid --max_seq_len 4096 \
  --calibration_device cuda \
  --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json \
  --calib_text_datasets wikitext-2 --calib_text_samples 8 \
  --prompt '<image>Describe this image in one sentence.' \
  --compile_only
```

Compilation also needs host RAM for graph transforms, QNN preparation and context
serialization. QNN preprocessing copies graph structure while sharing read-only
weight tensors; methods in one context reuse a single Python binary buffer.
These savings do not change the model's quantization or runtime graphs.
Stage cleanup also clears Dynamo export caches, collects Python objects and,
where glibc provides it, returns free heap pages with `malloc_trim`.

CUDA activation calibration rebuilds the PT2E graph after dead-code elimination
to discard unused FP32 attributes left by weight folding. Weight dequantization
and its constant casts/views are scheduled at their first consumer, rather than
materializing all FP32 weights before the first layer. Calibration retains FP32
arithmetic, the same quantization grids and the same samples. A regression covers
W4/W8 dequantization, shared weights, unused parameters and exact output equality.

On a 64 GB Linux host with systemd, run a 2B export inside a memory-limited user
scope after activating the environment and sourcing `env.sh`:

```bash
systemd-run --user --scope \
  -p MemoryMax=48G -p MemorySwapMax=0 \
  env MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python examples/qualcomm/oss_scripts/llama/llama.py \
    -a ./llama-output/qwen3_5-2b-m4096-hybrid-ar128 \
    --build_folder build-android --soc_model SM8750 \
    --decoder_model qwen3_5-2b \
    --source_safetensors /path/to/Qwen3.5-2B \
    --model_mode hybrid --max_seq_len 4096 --prefill_ar_len 128 \
    --calibration_device cuda --calibration_num_threads 4 \
    --calib_samples examples/qualcomm/oss_scripts/llama/assets/samples/vision.json \
    --calib_text_datasets wikitext-2 --calib_text_samples 8 \
    --prompt 'Describe this image.' --skip_qdq_save --compile_only
```

`MemoryMax` terminates the export if it exceeds 48 GiB, and `MemorySwapMax=0`
prevents this task from consuming swap. Adjust the limits to leave RAM for other
applications. A W4 deployment does not imply a W4-sized host compilation peak.

Local raw-text JSON/JSONL sources may replace `wikitext-2`. Image calibration
files may also include text-only samples with an empty `files` list. Generated
PTEs are `hybrid_llama_qnn.pte`, `tok_embedding_qnn.pte` and
`vision_encoder_qnn.pte`; `qwen35_export.json` records GPTQ targets and actual
cache encodings. Without `--skip_qdq_save`, `decode_qdq.pt2` retains full AR128
logits for analysis and `kv_qdq.pt2` retains the AR1 graph.

`--source_safetensors` supplies weights, tokenizer and encoder config from the
local directory. An explicit `--checkpoint` still overrides decoder weights.
No experiment scripts or previously generated PTEs are imported by this flow.

###### Validation record (2026-10-15)

The Qwen3.5/GDN experiment scripts, added validation tests, local logs, tensor
dumps, QDQ exports and PTEs were removed after integration. This section retains
the results; the full history of 52 reports is archived in
[Qwen3.5 test records](README_QWEN35_TEST_RECORDS.md). Historical commands and
artifact paths in that archive describe the original runs. Use `llama.py`
above to produce new artifacts.

| Verification | Recorded result |
|---|---|
| Final compatibility regression | 103 tests and 12 subtests passed |
| Pre-push regression (2026-10-15) | 15 existing tests and 3 subtests passed, including SWA calibration, quantized mask IO, cache handling, native ASAN/UBSAN checks and MHA-to-SHA conversion |
| Fresh normal-entry Qwen3.5-0.8B export | 4K context / AR128 compiled successfully; `kv_forward`, `prefill_forward` and `prefill_forward_256` each contain one QNN delegate with no CPU fallback |
| Fresh normal-entry Qwen3.5-2B image export | 4K context / AR128, two image samples and eight WikiText samples; all three decoder methods contain one QNN delegate with no CPU fallback; complete image inference passed on SM8850 |
| QNN compilation memory regression | 14 tests passed, including graph/state isolation, shared binary buffers across methods/partitions and MHA-to-SHA checks; a three-method quantized QNN compile/serialize smoke test preserved source weights and outputs |
| Covered models | Qwen3.5 and Lincal3.5 0.8B/2B export configuration, projection precision, calibration and cache layout; existing non-GDN model configuration checks |
| GDN/quantization | K/z W8, q/v/a/b W4, out_proj GPTQ; shared calibrated UINT16 cache encodings across AR128/AR1; native norms and quantized/float conversion |
| SWA | Separate window mask, fixed-length cache across AR changes, wraparound and partial updates; AR128 calibration compared with sequential decode, including a window smaller than AR |
| Generation graphs | Pure SWA skips FA256; mixed FA/SWA trims only full attention; padding preserves valid conv/recurrent state; selected-logit input preserves cache offsets |
| Prefill bounds | Final AR block checked against history capacity and physical RoPE positions, including non-aligned contexts and continuation |
| Native runtime | C++ cache tests passed with ASAN/UBSAN; Android `qnn_multimodal_runner` and `qnn_llama_runner` built successfully |
| Small Lincal PTE | SM8750, dim64, one GDN plus one SWA layer, AR128/AR1; 708,648 bytes; one QNN delegate per method, no CPU fallback; both masks and all public caches UINT16; decode compiled as `DeltaNetCoreVtcm` |
| Incremental op builds | Compiler/SDK/flag/source changes trigger rebuilds; identical settings reuse objects; normal export pins exact radix-16 reduction |

Final full-model measurements below are for **Qwen3.5-0.8B on SM8750**,
temperature 0, excluding model loading. Prefill was measured three times;
decode performance uses a fixed output budget with EOS ignored.

| Workload | Recorded result |
|---|---:|
| Text 128 input, 1 output; automatic FA256 | 85/87/86 ms; mean 86 ms, **1488.37 tokens/s** |
| Text 4096 input, 1 output; full FA4K | 3892/4084/4021 ms; mean 3999 ms, **1024.26 tokens/s** |
| Decode 16 input, output budget 496 | 495 measured decode forwards; **38.35 tokens/s** |
| Image/text 217 input, output budget 64 | First token **200.33 ms**, including about 30 ms vision encoding; subsequent decode **37.98 tokens/s** |
| Vision output SQNR against HF | Calibration cat 23.73 dB; held-out dog 17.88 dB; finite outputs |

WikiText-2 raw test, first 1024 tokens and 1023 prediction targets, with the
same new embedding PTE:

| Method | Device PPL | HF FP32 PPL |
|---|---:|---:|
| KV AR1 | **11.5479** | 10.8785 |
| Prefill AR128 | **11.6818** | 10.8785 |

These are recorded subset measurements, not full-dataset PPL or a complete
generation-quality assessment. The earlier 1684–1707 tokens/s result used a
different frozen short graph. An alternating three-round comparison measured
78 ms for that graph and 85.33 ms for the final graph, a 9.40% latency increase;
the accuracy fixes and calibration also differed. Full Lincal PPL, generation
and device performance have not been validated. The final regression and small
PTE compilation do not establish those full-model results.

Detailed historical kernel measurements are retained in the
[DeltaNet package README](../../custom_op/deltanet_op_package_htp/DeltaNetOpPackage/README.md).

##### Full Qwen3.5-2B export memory verification (2026-10-15)

The earlier 2B size-comparison script validated text generation and did not
compile or execute the 2B vision encoder. This run used the ordinary `llama.py`
entry point, the local Qwen3.5-2B checkpoint, CUDA calibration, two image samples,
eight WikiText-2 sequences of 1024 tokens, hybrid mode, context 4096 and AR128.
The default physical context capacity was 4224; `--skip_qdq_save` was enabled.

The preceding failed run ended after all decoder QNN preparation stages, without
a decoder PTE. The user reported a host freeze and reboot. No kernel OOM record
was available, so the reboot cause was not established as an OOM kill.

Two host allocation issues were fixed in `QnnBackend`: preprocessing now copies
graph structure and state mappings while sharing read-only tensor storage, and
each partition's shared context binary is converted to Python bytes once for all
methods. Temporary op-wrapper references and the NumPy binary are released before
packaging the shared bytes. For this run's 1,147,162,880-byte decoder context,
converting it once avoids two additional binary copies, about 2.14 GiB.

The retry ran serially in a user service with `MemoryMax=48G`,
`MemorySwapMax=0`, `CPUQuota=800%`, `MALLOC_ARENA_MAX=2`, and four CPU calibration
threads. Initial 44/46 GiB soft limits caused measurable memory-pressure stalls;
they were removed during final serialization. The documented command therefore
uses the hard memory limit without a soft limit.

| Export measurement | Result |
|---|---:|
| Exit status | 0 |
| Wall time, including diagnostic soft-limit stalls | 43 min 44.23 s |
| User-service memory peak, including charged cache | 50,437,963,776 bytes / 46.97 GiB |
| Process maximum RSS reported by `/usr/bin/time` | 49,478,300 KiB / 47.19 GiB |
| User-service swap peak | 0 bytes |
| Decoder PTE | 1,147,239,680 bytes / 1094.09 MiB |
| Token embedding PTE | 1,017,203,072 bytes / 970.08 MiB |
| Vision encoder PTE | 352,615,680 bytes / 336.28 MiB |

The cgroup and process RSS metrics have different accounting scopes. There is no
comparable peak measurement from the rebooted run, so these results do not
establish an end-to-end memory reduction percentage.

Device verification reused these SM8750 PTEs on SM8850 / PLK110,
serial `3B15AK00GLW00000`, with the V81 DeltaNet package and temperature 0.
Input was the local cow photograph plus `<image>请用中文描述这张图片。`.

| Single image/text generation measurement | Result |
|---|---:|
| Prompt / generated tokens | 216 / 172 |
| Vision execution | 95.86 ms |
| Decoder prefill execution, two AR128 blocks | 182.69 ms |
| Prompt evaluation / first token, including vision | 293 ms / 737.20 tokens/s |
| Decode | 32.75 tokens/s |

Generation reached EOS and correctly described the orange-shirted man, black
backpack and large white horned cow. This is a single generation check, not a
full-model PPL evaluation or an SM8850-native compilation benchmark. Local run
paths were `llama-output/qwen3_5-2b-m4096-hybrid-ar128-memoryfix`,
`run-qwen3_5-2b-memoryfix.log` and
`run-qwen3_5-2b-image-sm8850-memoryfix.log`; measurements are preserved here
independently of those generated files.

##### Qwen3.5-4B export attempt (2026-10-15, incomplete)

The local `/hunter/model/Qwen/Qwen3.5-4B` checkpoint is registered as
`qwen3_5-4b`. Its existing static config matches the checkpoint: 32 layers,
hidden size 2560, FFN size 9216, 16 linear key heads and 32 linear value heads.
The existing grouped GDN path handles the 2:1 value/key head ratio.

The attempt used SM8750, hybrid mode, context 4096, AR128, CUDA calibration,
two image samples and eight WikiText-2 sequences of 1024 tokens. Precision and
calibration remained the optimized Qwen3.5 recipe; no DSP kernel rewrite or
reduction in sample count was used. Only one full export ran at a time, inside
a service with `MemoryMax=48G`, `MemorySwapMax=0` and `CPUQuota=800%`.

The initial failure was CUDA OOM during quantized activation calibration.
PT2E retained unused FP32 attributes after folding weights, and placed weight
DQ ahead of its consumers. Rebuilding the graph after dead-code elimination
removed unused attributes; calibration now delays the constant DQ/cast/view
chains until their consumers. This reduced the measured quantized graph GPU
allocation from 20.57 GiB to 4.86 GiB. Both AR128 passes and all ten warm AR1
samples subsequently completed; 24 GDN out projections received fixed-grid
GPTQ and 1306 shared activation grids were calibrated on CUDA.

Host memory remained a separate blocker. Stage cleanup returned about 9.8 GiB
in the final attempt: RSS went from 39.59 GiB to 29.74 GiB after calibration.
The vision encoder compiled and wrote a 354,409,728-byte PTE. Embedding graph
preparation then completed, but the process was killed by OOM before its
context/PTE was written. The final service reported a peak of
49,708,736,512 bytes (46.30 GiB) and zero swap; earlier retries reached the
48 GiB hard limit. The final peak alone does not distinguish a cgroup-limit
kill from host-wide OOM.

| Component / check | Result |
|---|---|
| FP32 AR128, quantized AR128 and warm AR1 calibration | Passed |
| Vision encoder PTE | Generated |
| Token embedding PTE | Not generated |
| Hybrid decoder PTE / device inference | Not reached |
| Calibration-memory and QNN preprocessing regressions | 4 passed |

The calibration regressions cover W4/W8, shared weights, constant cast/view
chains, unused FP32 parameters, exact output equality and source-weight cache
release while a live exported graph remains usable. An eight-layer synthetic
GPU check measured 46.5 MiB before delayed DQ and 19.0 MiB after, with identical
outputs; this is a calibration-memory check, not an HTP performance result.

This directory is **not a complete runnable 4B export**:
`llama-output/qwen3_5-4b-m4096-hybrid-ar128-memoryfix`. The final log is
`run-qwen3_5-4b-memoryfix.log`, with per-attempt memory CSVs in that directory.
Full 4B export still requires a lower host-memory peak, for example by compiling
components in separate processes, or a host with more available RAM. No 4B
generation accuracy or inference performance is claimed by this attempt.

##### Qwen3.5-4B INT4 input embedding check (2026-10-15)

An isolated test quantized only the `248320 x 2560` input embedding table,
using the existing packed INT4 embedding kernel, group size 32 and FP32 scales.
The HQQ variant uses `hqq_scale_only`; RTN uses ordinary affine rounding.
Neither uses test-set activation calibration. The Transformers decoder, caches
and original tied LM head remained FP32; the quantized input table was passed
as `inputs_embeds` without replacing the shared floating-point parameters.

The AR128 test used the first 16,384 tokens of WikiText-2 raw **test** text,
split into 16 independent 1024-token windows, with 16,368 next-token targets.
Final hidden SQNR is measured after the last RMSNorm, before the LM head,
using the reference signal energy. This is a test subset, not full WikiText PPL.

| Input embedding | AR128 PPL | PPL change | Embedding SQNR | Final hidden SQNR |
|---|---:|---:|---:|---:|
| FP16 reference | 9.29720 | Baseline | Reference | Reference |
| INT4 g32, HQQ | 9.29632 | -0.0095% | 20.81 dB | 24.13 dB |
| INT4 g32, RTN | 9.29962 | +0.0260% | 20.22 dB | 23.98 dB |

On the same first 1024-token window, AR1 PPL was 6.87308 / 6.87516 / 7.00987
for FP16 / HQQ / RTN. HQQ increased PPL by 0.0302%, with final hidden SQNR
about 21.5 dB. AR128 on that same window gave effectively the same PPL and
hidden SQNR; the 16K and 1K aggregates above should not be used to infer an
encode/decode difference. HQQ's per-window PPL changes ranged from -1.77% to
+0.93%; the near-zero aggregate is not evidence that quantization improves accuracy.

The standalone HQQ PTE is **397,314,304 bytes / 378.91 MiB**, with shared
packed weights across AR1 and AR128. Table data is 68.75% smaller than the
theoretical FP16 payload of 1212.50 MiB; no successful FP16 4B embedding PTE
exists for a measured file-to-file comparison. Execution uses the ExecuTorch
CPU `embedding_4bit.dtype_out` kernel, not QNN/HTP.

On SM8850 / PLK110 (`3B15AK00GLW00000`), actual C++ PTE execution matched
PyTorch INT4 output bit-for-bit for all 16,384 AR128 tokens (41,943,040 FP32
elements) and all 1024 AR1 tokens (2,621,440 elements), maximum absolute error 0.
The local report and artifact are in
`llama-output/qwen3_5-4b-embedding-int4-eval/REPORT.md` and
`tok_embedding_int4_hqq.pte` in that directory.

These measurements support HQQ INT4 input embedding on this text subset.
They do **not** establish full quantized-decoder PPL, image accuracy or full
4B generation performance. The normal Qwen3.5 `llama.py` embedding export
path has not been changed to use this standalone PTE, and full 4B export has
not been restarted by this test.

##### Qwen3.5-0.8B earlier 1K embedding / KV ablation (2026-10-16)

These results use the separately exported `qwen3_5-0_8b-embedding-kv-ablation/kv16/hybrid_llama_qnn.pte`, not the original `qwen3_5-0_8b-m4096-hybrid-ar128` PTE (physical capacity 4224). The 61.91 tok/s baseline applies only to this 1K export. The original 4K image run on SM8850 recorded 46.7435 tok/s with 216 prompt tokens and 150 AR1 steps. Controlled context comparison: `llama-output/qwen3_5-0_8b-embedding-kv-ablation/context-baseline-check/REPORT.md`.

A subsequent controlled run on the same SM8750, raw text 16→1008, three rounds with the same runner, measured 61.78 tok/s for this 1K PTE and 39.51 tok/s for the original 4K PTE. The 4K PTE was reused unchanged. Single-block 128→1 prompt evaluation was 70 / 69 ms, respectively, since both used the short prefill bucket. No original-4K PPL/SQNR or INT4/UINT8 performance gain is established by the 1K ablation.

SM8750 / PKR110, hybrid AR128, logical 1K / physical capacity 1152. Decoder weights and calibrated grids were frozen: K/z W8, other Linear W4 group32, existing out-projection GPTQ. Only the six full-attention layers use UINT8 K/V; all 18 conv and 18 recurrent states remain UINT16. Pure-text inference; vision was calibrated and loaded but not executed.

PPL/SQNR use the first 8192 WikiText-2 raw test tokens, eight independent 1024-token windows (8184 targets), with the same Transformers FP32 reference. Hidden is measured after final RMSNorm and before LM head; it is not vocabulary-logit SQNR. Reference PPL: 22.90248. This is a subset evaluation.

| Embedding / FA KV | PPL | Final hidden SQNR vs FP32 | 128→1 prompt eval | 1024→1 prompt eval | 16→1008 decode |
|---|---:|---:|---:|---:|---:|
| FP16 embedding / KV16（1K消融基准） | 26.93968 | 11.485 dB | 69 ms | 573 ms | 61.91 tok/s |
| INT4 embedding / KV16 | 26.99978 | 11.369 dB | 72 ms | 577 ms | 63.06 tok/s |
| FP16 embedding / KV8 | 26.96509 | 11.442 dB | 72 ms | 584 ms | 67.64 tok/s |
| INT4 embedding / KV8 | 27.07849 | 11.319 dB | 72 ms | 587 ms | 67.19 tok/s |

The earlier KV8 rows also used an 8-bit MatMul RHS. They do not measure the
the later storage-only UINT8 experiment, whose attention MatMul remained 16-bit.

Performance is the median of three interleaved rounds, after warmup, using the production runner with temperature 0 and EOS ignored. Prompt evaluation includes embedding/cache work, excludes model loading/tokenization/vision, and excludes about 0–1 ms of final sampling/callback. Decode counts 1007 AR1 steps after the first token from prefill.

Embedding PTE shrank from 485.073 to 151.565 MiB (68.75% smaller); INT4 group32 HQQ lookup had 21.089 dB SQNR and exactly matched PyTorch packed lookup on-device. AR1 cache input buffers shrank from 23.332 to 16.588 MiB; this is not total process RSS.

A separate first-1K AR1 check gave PPL 11.69466 / 11.88040 / 11.81462 / 11.85262 in table order (FP32 10.87851). All four variants matched production generation logits code-for-code at positions 0/128/512/896 for both AR1 and AR128, after accounting for the trimmed prefill history layout.

The integration preserves packed embedding weights through a CPU portable PTE with `--embedding-quantize "4,32"`, uses that lookup during normal calibration, and preserves the original tied LM head. UINT8 FA cache selection remains experimental in this ablation, rather than changing the default recipe. Mixed GDN-UINT16 / FA-UINT8 cache allocation/updates and output requantization IO tagging were fixed. The already-compiled UINT8 QNN context was unchanged when correcting its PTE output dtype descriptors. Three Python regressions and the ASAN/UBSAN native cache regression passed.

The combined case added 0.515% PPL, reduced hidden SQNR by 0.166 dB, added 3 ms to AR128 prompt evaluation, and improved median decode speed by 8.53% on this device/context. Existing quantized-decoder error remains larger than these incremental changes; these data do not establish image accuracy or 4K/8K performance.

Local detailed report, configs and raw outputs: `llama-output/qwen3_5-0_8b-embedding-kv-ablation/REPORT.md`. Temporary scripts were removed after integration. Tests were run serially; each full export used a 36 GiB cgroup limit with no swap. That earlier experiment retained the then-current default precision; current recipe defaults are documented above.

##### Qwen3.5-0.8B 4K UINT8 storage / 16-bit compute ablation (2026-10-16)

All four variants were exported and evaluated on SM8750 / PKR110 / `431e3c37`.
Logical context is 4096, physical capacity 4224, prefill AR128. Only the six
full-attention layers change cache storage precision; all GDN conv/recurrent
states remain UINT16. In the KV8 cases, query quantization stays UINT16 and
MatMul RHS stays INT16, with calibrated cache conversions inside QNN.
These measurements use the retired storage-only implementation, distinct from the earlier
1K experiment's 8-bit attention compute.

Each variant was independently calibrated with the same two `vision.json`
image samples and eight WikiText-2 train samples, seed 42, using AR128 and warm
AR1. Accuracy uses the first 8192 WikiText-2 raw test tokens as two independent
4096-token windows, 8190 next-token targets. Transformers FP32 reference PPL
is 18.04003 / 18.04006 for AR128 / AR1. SQNR compares the final RMSNorm output,
before the LM head, against the matching FP32 hidden state.

| Embedding / FA KV storage | AR128 PPL | AR1 PPL | AR128 hidden SQNR | AR1 hidden SQNR | 128→1 prefill | 4096→1 prefill | 16→4080 decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 / UINT16 baseline | 21.1945 | 21.9655 | 11.145 dB | 10.498 dB | 74 ms | 3393 ms / 1207.19 tok/s | 38.68 tok/s |
| INT4 / UINT16 | 20.6700 | 21.5433 | 11.102 dB | 10.511 dB | 86 ms | 3468 ms / 1181.08 tok/s | 38.98 tok/s |
| FP16 / UINT8 | 21.1870 | 21.9863 | 11.132 dB | 10.501 dB | 74 ms | 5469 ms / 748.95 tok/s | 37.44 tok/s |
| INT4 / UINT8 | 20.7328 | 21.6102 | 11.080 dB | 10.481 dB | 76 ms | 5614 ms / 729.60 tok/s | 37.99 tok/s |

Performance is the median of three rounds, with warmup and alternating case
order, using production generation methods. These are pure-text measurements;
the vision PTE was loaded but not executed. Prefill timing includes embedding
and cache work, excludes loading/tokenization and final sampling/callback.
Decode measures 4079 AR1 calls; the first of 4080 outputs comes from prefill.
Round zero has Android Thermal Status 0 throughout; later rounds reach 1/2,
with `cdsp`/`cdsp_hw` cooling values still zero. DSP frequency was not locked,
so small decode differences do not establish a stable speedup. The short-prompt
INT4 result also varied from 74 to 87 ms across rounds.

Adding KV8 storage to INT4 embedding changes AR1 PPL by +0.0669 (+0.31%) and
hidden SQNR by −0.031 dB; AR128 changes are +0.0628 (+0.30%) and −0.022 dB.
Each variant adapts its calibration, so lower PPL in the INT4 rows does not
establish that embedding quantization itself improves accuracy. The existing
PTQ gap to FP32 remains. These subset results do not establish image quality.

INT4 embedding reduces its PTE from 485.073 to 151.565 MiB (−68.75%). The measured
storage-only KV8 version provides no end-to-end speed gain: relative to FP16/KV16, 4K
prefill latency rises 61.18% with KV8 alone and 65.46% with both changes.
The regression is already present in round zero without a thermal warning.
That storage-only version should not be treated as a performance improvement.

Detailed profiling of the production AR128 long-prefill decoder, with matched
synthetic warm caches and actual embedding inputs, measured HTP execution of
110.900 ms for KV16 versus 172.716 ms for KV8 (two warmups, three measured calls).
Twelve cache load conversions contribute 129.450 million aggregate cycles,
87.8% of the added aggregate cycles. Aggregate node cycles are not additive
wall-clock time. Compiler estimates show 36 MiB more spill for KV8 long prefill,
despite about 4.2% less estimated DDR read traffic for decode. These are static
estimates, not measured bandwidth counters. Small-graph VTCM residency does
not establish residency for the full 4K graph. Conversion placement and
temporary-cache lifetime remain performance issues in this implementation.

Production/diagnostic validation at six checkpoints found identical selected
logits in AR128/AR1 and identical cache codes in AR1. AR128 conv-cache differences
reach 118 codes with FP16 embedding and 290 with INT4, so complete prefill-cache
equivalence is not established. PPL/hidden are collected through diagnostic
methods; performance uses production methods.

The export also exposed a short-prefill trimming bug with the additional
storage-to-compute Q/DQ pair. `trim_prefill_capacity` now traces the full chain
and slices UINT8 history before expansion. Eight recipe/embedding regressions
passed, and an actual physical384/internal256 full-attention PTE matched Q/DQ
with AR128/AR1 output SQNR 75.56/74.27 dB; returned caches differed by at most
one code for AR128 and were identical for AR1.

This round recalibrates the original precision baseline; it does not report
accuracy deltas against the old `qwen3_5-0_8b-m4096-hybrid-ar128` artifact.
That old PTE measured 3505 ms for 4096→1 and 39.22 tok/s for 16→4080 in a
separate single run on this SM8750. The SM8850 image log's 46.743534 tok/s uses
another device and a shorter generation, and is not a baseline for this table.
Exports ran serially with a 36 GiB memory cap and no swap.

Detailed report, per-round results, prefix SQNR, calibration metadata, IO audits
and raw profiling events:
`llama-output/qwen3_5-0_8b-4k-embedding-kv-ablation-20261016/REPORT.md` and `summary.json`.

##### Qwen3-0.6B legacy KV8 execution path (2026-10-16)

An isolated export enabled `annotate_kv_8bit` for the local checkpoint
`/hunter/model/Qwen/Qwen3-0___6B`: KV mode, AR1, context1024, CUDA calibration
with eight WikiText-2 train sequences (8192 tokens), seed42, original SeqMSE50.
The normal Qwen3-0.6B recipe default was retained. The PTE is 652.319 MiB and
contains one QNN delegate, with 56 UINT8 cache inputs and 56 UINT8 cache outputs.
These input buffers occupy 55.945 MiB; this is not total process memory.

MHA→SHA produced 896 attention MatMuls (28 layers×16 query heads×2), all with
UINT16 left input, UINT8 right input and UINT16 output. All56 history-cache paths
stay UINT8 through per-head slicing, concat and reshape until MatMul. No history
UINT8→16bit expansion occurs. The672 explicit QNN Converts are UINT16→UINT8
on new-token tensors containing at most128 elements.

Actual device QHAS shows HVX
`q::ConvLayer.opt.convert_weights_to_signed.shuffled` producing QInt8 K/V with
the same scale and zero_offset0, followed by HMX `q::ConvLayer_s1.opt` MatMul
kernels producing QUInt16. There is internal unsigned-to-signed conversion and
layout preparation, but the RHS remains8bit. This differs from Qwen3.5's retired
`annotate_kv_8bit_io` implementation, which stored8bit and used a16bit RHS.

Device graph tracing and normal runner execution completed. The simple prompt
`What is 1+1? /no_think` generated repeated reasoning without a correct answer
before the128token limit. This does not establish usable quantized accuracy;
there was no matched KV16 control, PPL or hidden-SQNR evaluation in this check.
The PTE enables optrace, so its runtime timing is not a normal-performance result.

The check also repaired the `llama.py --optrace` path's obsolete `args.model`
references and migrated its `SimpleADB` construction to `QnnConfig`; the existing
masked-softmax device test's helper call now uses `soc_model`. The same compiled
PTE was reused to complete device tracing after the post-export failure.

Local PTE, calibrated Q/DQ graph, QNN tensor encodings, full optrace/QHAS
and detailed explanation: `llama-output/qwen3-0_6b-kv8-flow-20261016/REPORT.md`.

##### Qwen3.5-0.8B 4K native 8-bit K/V compute (2026-10-16)

The 0.8B recipe now uses `annotate_kv_8bit` for both FA cache storage and
attention MatMul RHS precision. The storage-only annotation and its cache
expansion branches were removed. New entries enter the calibrated 8-bit grid
before storage; historical entries remain 8-bit through concat and MatMul.
Query, softmax probability and MatMul outputs remain UINT16. GDN conv/recurrent
states retain UINT16. Re-exporting is required; existing PTEs are unchanged.
At the time of this measurement, 2B/4B retained FP16 embedding and KV16 defaults.
The subsequent 0.8B/2B performance-matrix integration changed Qwen 2B to
INT4 embedding/native KV8; 4B and Lincal 2B remain FP16/KV16.

This full export uses INT4/HQQ/group32 embedding, K/z and depthwise convolution
W8, other Linear weights W4/group32, and the existing out-projection GPTQ.
The new embedding PTE is byte-identical to the INT4/KV16 baseline. All 36 GDN
cache scale/zero-point pairs match that baseline. Prefill uses the existing
FP16 chunk64/leaf8 implementation; decode keeps the VTCM op.

SM8750 / PKR110 / `431e3c37`, logical context4096, physical capacity4224,
prefill AR128. Calibration uses the same two image/text samples and eight
WikiText-2 train sequences, seed42, CUDA. A single export ran with a36GiB memory
cap and no swap:38m14.56s elapsed,29.75GiB maximum RSS. All five decoder methods
(three production, two diagnostic) contain one QNN delegate without CPU
fallback. All60 FA MatMuls have UINT16×UINT8→UINT16 tensors; there are no history
cache8→16 Convert nodes. Production cache IO has12 UINT8 FA tensors and36
UINT16 GDN tensors in each direction.

Accuracy uses the same first8192 WikiText-2 test tokens as the previous4K
experiment: two independent4096-token windows,8190 next-token targets, and the
same Transformers FP32 reference. SQNR measures final RMSNorm hidden before the
LM head. Each precision variant was calibrated separately; this compares full
export pipelines rather than replacing one grid with every other grid frozen.
Baseline accuracy below reuses the preceding matched evaluation.

| INT4 embedding configuration | AR128 PPL | AR1 PPL | AR128 hidden SQNR | AR1 hidden SQNR |
|---|---:|---:|---:|---:|
| KV16 | 20.6700 | 21.5433 | 11.102 dB | 10.511 dB |
| Retired: KV8 storage /16-bit RHS | 20.7328 | 21.6102 | 11.080 dB | 10.481 dB |
| Current: KV8 storage /8-bit RHS | 20.7568 | 21.5803 | 11.066 dB | 10.474 dB |

Relative to INT4/KV16, native KV8 changes AR128 PPL by+0.42% and hidden SQNR
by−0.037dB; AR1 PPL changes by+0.17% and hidden SQNR by−0.038dB. HF FP32 PPL
is18.04; the existing PTQ gap remains separate from the small additional KV8
error. Six production/diagnostic checkpoints have identical selected logits
in both modes and identical caches in AR1. AR128 conv-cache differences reach
290 codes, as in the INT4 baseline, so complete prefill-cache equivalence is
not established.

Performance below was collected anew for both models, three alternating
rounds with the same production runner and warmup, taking the median. Inputs
are raw text tokens; vision is loaded but not executed. Timing includes
embedding/cache work and excludes loading and tokenization.

| Configuration | 128→1 prefill | 4096→1 prefill | 16→4080 decode |
|---|---:|---:|---:|
| INT4/KV16 | 79 ms /1620.25 tok/s | 3467 ms /1181.42 tok/s | 38.95 tok/s |
| INT4/KV8,8-bit RHS | 78 ms /1641.03 tok/s | 3245 ms /1262.25 tok/s | 61.70 tok/s |

Decode throughput rises58.40% and4K prefill latency falls6.40%. The1ms change
in128-token prefill is within run variation. Decode counts4079 AR1 steps; the
first of4080 output tokens comes from prefill. KV16 decode runs span
38.74–39.50tok/s, KV8 runs60.82–61.90tok/s. Android thermal statuses0/1/2 were
observed without a frequency lock; snapshots are retained for every run.
The retired storage-only result5614ms/37.99tok/s is historical, not part of
this fresh paired timing batch.

All cache input buffers for `kv_forward` shrink from59.332MiB to34.588MiB
(FA halves; GDN stays unchanged). These are buffer sizes, not total process
memory. QNN compiler estimates for decode spill/fill fall from26.80/27.86MiB
to1.75/2.80MiB, estimated reads from554.65MiB to492.47MiB and writes from95.86MiB
to21.31MiB. These are static estimates, not measured hardware bandwidth.

Eight recipe/embedding regressions passed, covering mixed cache types,
AR128/AR1, short-prefill trimming and unchanged larger-model defaults.
Black and `git diff --check` passed. The ordinary `llama.py --pre_gen_pte` path
also completed a23-token Chinese arithmetic prompt, producing `1+1=2` and
stopping at EOS after padded AR128 prefill. This is a runner integration check,
not a general answer-quality evaluation. PTEs, paired timings, accuracy, prefix SQNR,
calibrated encodings and tensor audits:
`llama-output/qwen3_5-0_8b-4k-kv8-compute-20261016/REPORT.md` and `summary.json`.

##### Qwen3.5-0.8B complete 4K deployment PTE recheck (2026-10-16)

A deployment bundle is available at
`llama-output/qwen3_5-0_8b-m4096-hybrid-ar128-e4-kv8/`.
It contains the complete24-layer decoder, INT4 embedding, vision encoder and tokenizer.
This package reuses the preceding calibrated production methods and QNN context;
only the two diagnostic PTE entrypoints were removed. All116 metadata methods are
preserved. Production method signatures and QNN payload hashes were checked.
The shared QNN context still contains its original internal graphs, so its size
is nearly unchanged. This is packaging, without new calibration or compilation.

The actual packaged PTE was pushed to SM8750/431e3c37 and checked by SHA256,
then measured afresh for three rounds with the same normal production runner.
Logical context4096, physical capacity4224, AR128; FA K/V storage and RHS UINT8,
GDN conv/recurrent UINT16. Vision is loaded but not executed.

| Test | Median time | Median throughput |
|---|---:|---:|
| 128 → 1 prefill | 75 ms | 1706.67 token/s |
| 4096 → 1 prefill | 3215 ms | 1274.03 token/s |
| 16 → 4080 decode | 65664 ms | 62.12 token/s |

Raw times:128-token prefill94/74/75ms;4K prefill3259/3181/3215ms;
long decode65664/65001/66122ms. All thermal snapshots reported status0;
DSP frequency was not locked. Each process runs16→32 warmup before its measured
generation. Timing includes embedding/cache work and excludes loading,
tokenization and vision. Long decode generates4080 outputs with4079 AR1 steps;
the first output comes from prefill.4K prefill processes32 AR128 blocks.

The deployment README contains the ordinary `llama.py --pre_gen_pte` command.
Complete raw logs, configs, counts, thermal snapshots and median calculations:
`llama-output/qwen3_5-0_8b-m4096-hybrid-ar128-e4-kv8/BENCHMARK.md` and
`benchmark_summary.json`. Accuracy remains the preceding identical-production-graph
evaluation; no new PPL/SQNR is claimed for this performance recheck.

##### Qwen3.5-family: remove short-context prefill graphs (2026-10-16)

The exporter no longer clones or compiles `prefill_forward_256`, and the runner
no longer discovers, loads or selects short-context buckets. The additional
model plumbing and optrace bucket targets were removed. Hybrid decoder PTEs
now contain only `kv_forward` and `prefill_forward` inference methods plus their
metadata; KV mode retains only `kv_forward`. Embedding and encoder methods are
unchanged. Lincal SWA masks, cache update logic and logical-capacity trimming
remain in place. AR128 still processes4096 prompt tokens in32 calls.

A full0.8B export was recalibrated on CUDA and recompiled from the local model
with the same seed/data/precision. Actual PTE methods and QNN context graph
names were both checked: exactly `kv_forward` and `prefill_forward`, with no
short-context or diagnostic graph. All metadata and GPTQ statistics match the
preceding export; embedding PTE bytes match. This is not a bytewise comparison
of all decoder weights. All48 cache encodings match; FA remains UINT8 and GDN
state UINT16. Decoder size is458.477MiB. Export took16m28.42s with17.04GiB
maximum RSS, under a36GiB cap with no swap.

Using newly built Android runners, the new PTE completed three rounds on
SM8750/431e3c37, with all thermal snapshots0 and no frequency lock:

| Test | Previous bundle median | New two-graph PTE median | New throughput |
|---|---:|---:|---:|
| 128→1 prefill | 75ms | 104ms | 1230.77tok/s |
| 4096→1 prefill | 3215ms | 3225ms | 1270.08tok/s |
| 16→4080 decode | 65664ms | 64202ms | 63.53tok/s |

The previous figures are historical measurements, not fresh paired controls.
All short prompts now execute the full-context graph, so128-token prefill
latency increases38.7%;4K prefill is essentially unchanged. The small decode
difference is not attributed solely to bucket removal. Raw new times are
103/104/104ms,3225/3158/3237ms and63752/64873/64202ms, respectively.
Decode counts4079 AR1 steps; the first output comes from prefill. Timing includes
embedding/cache work and excludes loading/tokenization/vision execution.

Eight recipe/embedding regressions passed, both Android runner builds passed,
and the changed Python files passed serial Black checks. The ordinary
`llama.py --pre_gen_pte` path answered the23-token Chinese arithmetic prompt
with `1+1=2`, stopped at EOS and logged `method=prefill_forward`.
No new PPL/SQNR evaluation was performed in this change.

New complete deployment bundle, ordinary run command, raw measurements and
PTE/QNN audits:
`llama-output/qwen3_5-0_8b-m4096-hybrid-ar128-e4-kv8-no-bucket/README.md`,
`BENCHMARK.md` and `deployment_manifest.json`.
Earlier measurements in this README retain their historical graph selection;
they do not describe the new single-prefill configuration.

##### Re-exporting after the masked-softmax causality fix

Models configured with `masked_softmax=True`, including `lincal-1_7b`, now
exclude masked scores before the row minimum and explicitly zero masked
probabilities after softmax. This removes the dependency on future keys and
masked values in the saved QDQ graph. The same implementation is used for
calibration, prefill and decode, without changing the KV layout or prefill AR.
The finite fill avoids introducing infinity into the QNN graph.

For R3 models, MHA-to-SHA conversion keeps native Hadamard rotations batched
and splits their outputs into attention heads. This avoids the QNN 2.49
`q::ConvLayer_s1.opt` preparation failure observed when splitting the rotations
themselves. R3, per-head attention and the KV interface remain enabled/unchanged.

Existing PT2/PTE files do not pick up Python source changes. Re-run the normal
`llama.py` calibration and export with the original SafeTensors checkpoint and
a **new artifact directory**. No weight retraining is needed; recalibration
collects activation encodings for the changed graph. Omit `--skip_qdq_save` when
both `decode_qdq.pt2` and PTE are needed. Do not copy old graph encodings onto the
new topology or use the historical PT2-only diagnostic patches as a release.

Validate the new artifacts through [model_validation](model_validation/README.md).
The PT2 future-token causality guard remains enabled; passing small regression
models does not establish full-model PPL or answer quality.

##### Mixed Chinese/English raw-text calibration

For text-only models, add these options to the export command to calibrate on
CCI3-HQ and WikiText-2:

```bash
--calib_text_datasets cci3-hq wikitext-2 \
--calib_text_samples 128 128 \
--calibration_device cuda
```

`cci3-hq` maps to [BAAI/CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ);
`wikitext-2` maps to the `wikitext-2-raw-v1` configuration of
[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext).
Both use the training split and streaming reads with a bounded shuffle buffer,
not a full-corpus download. `--seed` controls shuffling and long-document crops.
Counts are **token sequences per source**, up to `max_context_len` tokens each,
not document counts or deployment prefill AR. One count applies to all sources;
multiple counts specify the mix (for example `192 64`). Short documents are
packed with EOS separators; long documents contribute a randomly cropped window.
Empty rows are skipped; an empty source fails, and a short source logs a warning.

CCI3-HQ requires access approval on its dataset page and an authenticated local
Hugging Face client. A permission error is not silently replaced with English-only
calibration. Already downloaded data can be supplied as local JSON/JSONL files
containing `{"text": "..."}` records instead of `cci3-hq`.
`--calib_samples` can also be included for representative chat examples.
Changing calibration data requires exporting a new PTE; use a new artifact directory
to retain the previous model for comparison. It does not change deployment AR.

##### Preserve complete thinking conversations

For preformatted conversations, add the local source path to both
`--calib_text_datasets` and `--calib_text_preserve_sources`. Each nonempty `text`
row then becomes one sequence, with no chat-template rendering, concatenation,
truncation or automatic EOS insertion. Rows outside 2..`max_context_len` tokens
raise an error. Include the model's role/EOS markers in the source text yourself.
Unlisted sources, including CCI3-HQ and WikiText, keep their existing behavior.
`--calib_text_samples` limits rows for preserved sources and packed sequences for
other sources. Use the full row count to include the whole curated file.

See [thinking calibration preparation](model_validation/calibration/README.md)
for the reusable curator, GPU candidate generation and token-exact loader
validation. The local Luxi v1 bundle at
`assets/samples/luxi_think_calibration_v1/` contains 256 Chinese R1 samples and
30 reviewed Luxi two-turn conversations. This bundle is not tracked in Git;
copy it separately or prepare a new bundle using the linked instructions.
Use its ChatML JSONL through this preserved-source path; `messages.json` is for
inspection, not the recommended calibration input. Calibration is not supervised
fine-tuning and does not establish an accuracy improvement until a new PTE is evaluated.

##### Runtime repetition penalty

Both `llama.py` and the QNN text/multimodal runners accept
`--repetition_penalty 1.1`. The default `1.0` disables the penalty and preserves
existing sampling. Values must be finite and at least `1.0`.
The penalty is applied once per distinct token in the most recent 256 token
positions of the prompt and accepted output, including thinking and earlier turns
in the same runner conversation. Repeated occurrences occupy separate positions
but are penalized only once; a token stops being penalized when its last occurrence
leaves the window. Lookahead candidates do not enter history until accepted.
The window slides independently of SWA and is cleared when the runner is reset.
The text runner also accepts a constructor-level `repeat_last_n` (default 256).
The identity-test driver's JSON exposes it as `"repeat_last_n": 64`; use a positive
32-bit integer. This changes the actual history window, not just the log label.
With `repetition_penalty=1.0`, the penalty is disabled regardless of window size.
For seen tokens, positive logits are divided by the penalty and negative logits
are multiplied by it, before temperature/top-p sampling. Quantized logits are
dequantized using the PTE's `get_logits_scale` and `get_logits_zero_point` metadata
when the penalty is enabled. Older quantized PTEs missing these metadata methods
must be re-exported to use the penalty.

Rebuild and update the runner (or Android AAR) once to use the 256-token window;
existing binaries keep their old behavior. Subsequent changes to the penalty
strength require neither runner rebuilding nor PTE export. EOS stopping remains
independent (`--ignore_eos=false` on the native runner). Repetition penalty can
reduce loops, but is not a substitute for validating model/quantization accuracy.

##### Runtime input logs

The QNN text runner (including JNI and the identity-test driver) logs each call's
effective `temperature`, `repetition_penalty`, `repeat_last_n`, and `ignore_eos`
as `[QNN_GENERATION]`. It also logs the effective sequence limit and prefill size.
`[QNN_INPUT]` identifies the cached token count, pending output token (`-1` means
none), and explicit input BOS/EOS counts. Later turns reuse KV history; they do
not resend that history as text.

`[QNN_PROMPT]` contains the complete string passed to the tokenizer, including
ChatML and thinking markers, independent of `echo`. Chunks have UTF-8 byte offsets
and JSON-escaped `text` strings (`\n` means a newline); concatenate the decoded
strings to recover the exact input. Chunking avoids the native log's 256-byte
buffer limit. For tokenized-file input this string is the **file path**, not text
sent to the tokenizer. `max_new_tokens=0` skips input logging and inference.

These Info logs contain user/system prompt content; treat them as sensitive and
disable Info logging in production builds that must not retain prompts. Rebuild
and deploy the runner/driver or AAR to enable these logs; no PTE export is needed.

##### Custom calibration samples (`--calib_samples`)

`--calib_samples` accepts one or more JSON files. Each file is a flat list of sample objects. Each sample has a `messages` field following the HuggingFace chat template, and an optional `files` field for media inputs (local paths or URLs):

```json
[
  {
    "files": ["path/or/url/to/files"],
    "messages": [
      {"role": "user",    "content": "..." },
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

`files` is only required for multimodal models (VLM: image paths/URLs, ALM: audio paths/URLs). For LLM-only models, `files` can be omitted. `content` can be a plain string or a list of HuggingFace content blocks (e.g. `[{"type": "image"}, {"type": "text", "text": "..."}]` for vision inputs).

Ready-to-use examples for each model type are provided under `assets/samples/`:

| Model type | Example file |
|---|---|
| LLM | [assets/samples/text.json](assets/samples/text.json) |
| ALM (audio) | [assets/samples/audio.json](assets/samples/audio.json) |
| VLM (vision) | [assets/samples/vision.json](assets/samples/vision.json) |

For the full quantization workflow: PTQ (Post Training Quantization), QAT(Quantization Aware Training / Distillation), Mixed-precision, Quantization error analysis, see [quantization_guidance.md](quantization_guidance.md).

#### Use attention sink for multi-turn conversations
Attention sink is a way to evict cache when maximum context length be reached.
There are two mainly concept for attention sink:

1. **Maintain Attention Sinks**: Always include several initial tokens as attention sinks in the kv cache.
2. **Redefine Positional Context**: Use positions relative to the cache instead of absolute positions from the original text, enhancing relevance and coherence in generated responses.

<figure>
    <img src="assets/AttentionSinkFeature.png" alt="Attention Sink Feature">
    <figcaption>This figure shows how the attention sink operates for the kv cache in LLMs when `max_context_len = 8`, `sink_size = 4`, and `eviction_batch_size = 2`. The yellow blocks represent sink tokens, blue blocks indicate the remaining tokens, and the red blocks show the newly generated token.
    </figcaption>
</figure>

This feature supports fluent multi-turn conversations and manages long-context scenarios. To enable it, set `--use_attention_sink <sink_size>,<batch_eviction_size>`.

##### Explanation of Parameters Related to Attention Sink:
1. **`--max_seq_len`**: Maximum sequence length the model can generate
2. **`--max_context_len`**: Maximum length of the model's memory/cache, including both prompt tokens and generated tokens
3. **`<sink_size>`**: Always include `sink_size` initial tokens as attention sinks in the kv cache.
4. **`<batch_eviction_size>`**: How many tokens to evict from the cache at once when the cache is full.

Example:
```bash
# Compile llama pte file and attention sink evictor pte file with sink_size = 4 and batch_eviction_size = 64
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 4096 --max_context_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --calib_tasks wikitext --calib_limit 1 --use_attention_sink 4,64 --compile_only
```

After running this, the `attention_sink_evictor.pte` file will be generated in the artifacts directory. This file is necessary for using the attention sink feature, as it handles removing the `eviction_batch_size` tokens from the kv cache, retaining the first `sink_size` tokens, and re-rotating the remaining tokens in the kv cache.

For multi-turn conversations or scenarios with long context using attention sink, you can set max_seq_len higher than the max_context_len used during compilation:
```bash
# Run llama with attention sink in multi-turn conversation scenario
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 4096 --prompt "I would like to learn python, could you teach me with a simple example?" "Could you give more difficult example in python?" "Could you add a GUI for this game?" "Could you tell me more about tkinter?" "Is possible to deploy on website?" ---pre_gen_pte ${PATH_TO_ARTIFACT_IN_1ST_RUN}  --use_attention_sink 4,64
```

If you want to modify `sink_size` or `batch_eviction_size`, or if you have a pre-compiled llm pte file and wish to use the attention sink feature, you can recompile the `attention_sink_evictor.pte` with different attention sink config.

```bash
# Compile attention sink evictor pte file with sink_size = 4 and batch_eviction_size = 128
python examples/qualcomm/oss_scripts/llama/llama.py --build_folder build-android --device ${SERIAL_NUM} --soc_model ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 4096 --prompt "I would like to learn python, could you teach me with a simple example?" "Could you give more difficult example in python?" "Could you add a GUI for this game?" "Could you tell me more about tkinter?" "Is possible to deploy on website?" ---pre_gen_pte ${PATH_TO_ARTIFACT_IN_1ST_RUN}  --use_attention_sink 4,128
```

Please make sure to use the same `--max_context_len`, `--prefill_ar_len`, and `--model_mode`, etc., as those used in the LLM to ensure the kv cache shape is correct.

<!-- QWEN35_CONTEXT_MATRIX_20261016_BEGIN -->
##### Qwen3.5 context performance: 0.8B / 2B (2026-10-17)

SM8750, text only, AR128, three-run medians. Both models use INT4/group32 HQQ embedding and native UINT8 full-attention K/V; GDN states remain UINT16. K/z and depthwise convolution use W8; other Linear layers use W4/group32 with the existing out_proj GPTQ. Decoder contexts contain only `kv_forward` and `prefill_forward`.

| 上下文 token | 0.8B Prefill token/s | 0.8B Decode token/s | 2B Prefill token/s | 2B Decode token/s |
|---:|---:|---:|---:|---:|
| 1024 | 1726.81 | 67.35 | 1196.26 | 35.25 |
| 2048 | 1605.02 | 63.36 | 1130.87 | 34.98 |
| 3072 | 1455.23 | 64.74 | 1042.77 | 36.23 |
| 4096 | 1261.86 | 63.51 | 926.70 | 35.05 |
| 8192 | 1293.75 | 39.97 | 773.63 | 27.43 |

Prefill: n input tokens / 1 output. Decode: 16 input / (n−16) output, timed across the actual n−17 AR1 calls. Physical capacity is n+128. Temperature is 0; loading, tokenization and vision execution are excluded. Every measured run started at thermal status 0; some sustained decode runs ended at 1/2. No frequency lock was used.

The matrix used a local 2B recipe override. After measurement, the same configuration was integrated into the standard `llama.py` entry point for 0.8B/2B. This matrix measures performance; PPL/SQNR were not rerun. Full definitions and results are in [Qwen3.5 test records](README_QWEN35_TEST_RECORDS.md). PTEs, logs, CSVs and plots are under `llama-output/qwen35-context-performance-20261016/`.
<!-- QWEN35_CONTEXT_MATRIX_20261016_END -->
