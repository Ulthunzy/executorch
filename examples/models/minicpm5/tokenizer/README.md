# MiniCPM5 tokenizer override

MiniCPM5 requires different handling for a regex `Split` pre-tokenizer when
`invert` is false. `pre_tokenizer.cpp` is the complete MiniCPM5-specific
implementation. The rest of the tokenizer sources continue to come from the
pinned `meta-pytorch/tokenizers` submodule.

Enable the implementation when configuring the ExecuTorch libraries used by
MiniCPM5:

```sh
cmake -B cmake-out/minicpm5 \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_USE_MINICPM5_TOKENIZER=ON
cmake --build cmake-out/minicpm5 --parallel
```

The option replaces only `src/pre_tokenizer.cpp` in the `tokenizers` target for
that build. It applies to every runner linked against that target, not only to
MiniCPM5 selected at runtime. Use separate builds for MiniCPM5 and other models.

For the Qualcomm Android runner:

```sh
# MiniCPM5: build-android-minicpm5/
bash backends/qualcomm/scripts/build.sh --skip_x86_64 --use_minicpm5_tokenizer

# Other models: build-android/
bash backends/qualcomm/scripts/build.sh --skip_x86_64
```

The script explicitly sets the option to ON or OFF on every invocation, including
with `--no_clean`. MiniCPM5 uses `build-android-minicpm5`, `build-x86-minicpm5`, and
`build-oe-linux-minicpm5`; ordinary builds retain the original directory names.
Create a build first before using `--no_clean` for that directory. When running
MiniCPM5 with `llama.py`, pass `--build_folder build-android-minicpm5`.

Existing binaries are not changed by updating this script. If an old
`build-android` was built with the MiniCPM5 option enabled, rebuild it using the
ordinary command above before using it for other models. Previously exported
PTE files do not need to be exported again for this build-directory change.

For manual CMake invocations, the option defaults to OFF only in a fresh build
directory. CMake retains cached options: explicitly pass
`-DEXECUTORCH_USE_MINICPM5_TOKENIZER=OFF` when returning an existing build to the
upstream implementation, then rebuild its libraries and runners.
