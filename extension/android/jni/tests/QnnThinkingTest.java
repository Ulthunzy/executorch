/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmModuleConfig;

/** Uses early.pte/loop.pte from test_thinking_device.py, not a language-quality test. */
public final class QnnThinkingTest {
  private static final String PROMPT =
      "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n";

  private static final class Output implements LlmCallback {
    final StringBuilder text = new StringBuilder();
    LlmModule stopAtClose;
    public void onResult(String piece) {
      text.append(piece);
      if (stopAtClose != null && text.toString().endsWith("</think>")) {
        stopAtClose.stop();
      }
    }
    public void onStats(String stats) {}
    public void onError(int code, String message) {
      throw new AssertionError(code + ": " + message);
    }
  }

  private static void check(boolean value, String message) {
    if (!value) throw new AssertionError(message);
  }

  private static LlmGenerationConfig config(boolean repair, boolean budget, int count) {
    return LlmGenerationConfig.create().echo(false).temperature(0f).seqLen(96)
        .maxNewTokens(count).repairThinkEos(repair).enableThinkBudget(budget)
        .thinkBudget(2).numBos(0).numEos(0).build();
  }

  private static String generate(LlmModule module, String prompt, boolean repair, boolean budget) {
    Output out = new Output();
    check(module.generate(prompt, config(repair, budget, 20), out) == 0, "generate failed");
    return out.text.toString();
  }

  public static void main(String[] args) {
    check(args.length == 3, "EARLY.pte LOOP.pte TOKENIZER.json");
    LlmGenerationConfig defaults = LlmGenerationConfig.create().echo(false).seqLen(96).build();
    check(defaults.isRepairThinkEos() && defaults.isEnableThinkBudget(), "protections not on by default");
    check(defaults.getThinkBudget() == 512, "wrong default think budget");
    LlmModuleConfig moduleDefaults = LlmModuleConfig.create()
        .modulePath(args[0]).tokenizerPath(args[2])
        .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA).qnnDecoderModel("lincal")
        .temperature(0f).build();
    check(moduleDefaults.getQnnRepetitionPenalty() == 1.05f, "wrong default penalty");
    check(moduleDefaults.getQnnRepeatLastN() == 128, "wrong default window");
    try (LlmModule module = new LlmModule(moduleDefaults)) {
      for (int reload = 0; reload < 2; reload++) {
        Output out = new Output();
        check(module.generate(PROMPT, defaults, out) == 0, "default generate failed");
        check(out.text.toString().contains("</think>OK"), "default EOS repair not effective");
        module.unload();
      }
    }
    try (LlmModule legacy = new LlmModule(4, args[0], args[2], "", "lincal", 0f, null)) {
      Output out = new Output();
      check(legacy.generate(PROMPT, 96, out, false) == 0, "legacy default generate failed");
      check(out.text.toString().contains("</think>OK"), "legacy default repair not effective");
      legacy.resetContext();
      Output chat = new Output();
      check(legacy.generateWithChat("你好", 20, chat, false) == 0, "legacy chat failed");
      check(chat.text.toString().contains("</think>OK"), "legacy chat repair not effective");
    }
    for (int mode : new int[] {0, 1}) {
      for (boolean loop : new boolean[] {false, true}) {
        try (LlmModule module = new LlmModule(LlmModuleConfig.create()
            .modulePath(args[loop ? 1 : 0]).tokenizerPath(args[2])
            .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA).qnnDecoderModel("lincal")
            .qnnEvalMode(mode).qnnRepetitionPenalty(1f).qnnRepeatLastN(64).temperature(0f).build())) {
          for (boolean repair : new boolean[] {false, true}) {
            for (boolean budget : new boolean[] {false, true}) {
              module.resetContext();
              String first = generate(module, PROMPT, repair, budget);
              boolean closed = loop ? budget : repair;
              check(first.contains("</think>OK") == closed, "switch not applied: " + first);
              String second = generate(module, PROMPT, !repair, !budget);
              check(second.contains("</think>OK") != closed, "per-call config leaked: " + second);
            }
          }
          module.resetContext();
          Output stopped = new Output();
          stopped.stopAtClose = module;
          check(module.generate(PROMPT, config(true, true, 20), stopped) == 0, "stop failed");
          check(stopped.text.toString().endsWith("</think>"), "did not stop at closing token");
          check(!stopped.text.toString().contains("OK"), "generated after cancellation");
          String resumed = generate(module, PROMPT, true, true);
          check(resumed.endsWith("</think>OK<|im_end|>"), "pending close/KV continuation failed");
          module.resetContext();
          Output zero = new Output();
          check(module.generate(PROMPT, config(true, true, 0), zero) == 0, "zero budget failed");
          check(zero.text.length() == 0, "zero budget produced output");
          if (!loop) {
            String firstToken = generate(module, PROMPT + "<think>", true, false);
            check(firstToken.equals("</think>OK<|im_end|>"), "first prefill EOS was not repaired");
          }
        }
      }
    }
    System.out.println("PASS: JNI four combinations, per-call toggles, first token, stop and KV continuation");
    System.exit(0);
  }
}
