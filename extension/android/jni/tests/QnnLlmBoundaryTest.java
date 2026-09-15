/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.json.JSONObject;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmModuleConfig;

public final class QnnLlmBoundaryTest {
  private static final class Output implements LlmCallback {
    int error;
    String stats;
    public void onResult(String text) {}
    public void onStats(String value) { stats = value; }
    public void onError(int code, String message) { error = code; }
  }

  private static void check(boolean value, String message) {
    if (!value) throw new AssertionError(message);
  }

  private static LlmModule module(String pte, String tokenizer, String model) {
    return new LlmModule(LlmModuleConfig.create().modulePath(pte).tokenizerPath(tokenizer)
        .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA).qnnDecoderModel(model)
        .qnnEvalMode(LlmModuleConfig.QNN_EVAL_MODE_HYBRID).temperature(0.0f).build());
  }

  private static LlmGenerationConfig config(int bos, int eos) {
    return LlmGenerationConfig.create().echo(false).warming(true).maxNewTokens(1)
        .ignoreEos(true).numBos(bos).numEos(eos).build();
  }

  public static void main(String[] args) throws Exception {
    check(args.length == 4, "AR16.pte AR4.pte QWEN_TOKENIZER.json LLAMA3_TOKENIZER.json");
    try (LlmModule full = module(args[0], args[2], "qwen3")) {
      Output result = new Output();
      full.generate("你好", config(0, 0), result);
      check(result.error == 0, "full-context prefill failed");
      check(new JSONObject(result.stats).getInt("generated_tokens") == 1, "missing token");
      Output second = new Output();
      check(full.generate("你好", config(0, 0), second) == 0x12,
          "BERT-style prefill must require reset between calls");
      check(second.error == 0x12, "expected InvalidArgument, not abort");
      full.resetContext();
      full.generate("你好", config(0, 0), new Output());
    }
    try (LlmModule chunked = module(args[1], args[2], "qwen3")) {
      StringBuilder text = new StringBuilder();
      for (int i = 0; i < 12; ++i) text.append("<|im_start|>");
      chunked.generate(text.toString(), config(0, 0), new Output());
      chunked.resetContext();
      Output oversized = new Output();
      check(chunked.generate(text + "<|im_start|>", config(0, 0), oversized) == 0x12,
          "chunked prefill must reject capacity overflow");
      check(oversized.error == 0x12, "expected InvalidArgument");
      chunked.generate("你好", config(0, 2), new Output());
    }
    try (LlmModule llama = module(args[0], args[3], "llama3")) {
      Output plain = new Output();
      llama.generate("Hello", config(0, 0), plain);
      llama.resetContext();
      Output added = new Output();
      llama.generate("Hello", config(2, 2), added);
      check(new JSONObject(added.stats).getInt("prompt_tokens") ==
          new JSONObject(plain.stats).getInt("prompt_tokens") + 4,
          "BOS/EOS counts were treated as a boolean switch");
    }
    System.out.println("PASS: hybrid full/chunked capacity, reset, BOS/EOS counts");
    System.exit(0);
  }
}
