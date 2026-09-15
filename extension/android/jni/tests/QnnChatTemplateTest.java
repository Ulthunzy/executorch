/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmModuleConfig;

/** Compare the actual echoed runner input with the identity-evaluation ChatML template. */
public final class QnnChatTemplateTest {
  private static final String FIRST =
      "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n";
  private static final String SECOND =
      "\n<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n";

  private static final class Output implements LlmCallback {
    String echoed;
    final StringBuilder generated = new StringBuilder();
    LlmModule stopAtEcho;

    public void onResult(String text) {
      if (echoed == null) {
        echoed = text;
        if (stopAtEcho != null) stopAtEcho.stop();
      } else {
        generated.append(text);
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

  private static void chat(LlmModule module, String prompt, String expected) {
    Output out = new Output();
    check(module.generateWithChat(prompt, 20, out, true) == 0, "chat failed");
    check(expected.equals(out.echoed), "template mismatch: " + out.echoed);
    check(out.generated.toString().equals("<think></think>OK<|im_end|>"),
        "thinking protection changed: " + out.generated);
  }

  private static void checkConversation(LlmModule module, String first) {
    Output empty = new Output();
    check(module.generateWithChat("", 20, empty, true) == 0, "empty input failed");
    check(empty.echoed == null, "empty input changed history");
    check(module.generateWithChat("你好", 0, empty, true) == 0, "zero budget failed");
    check(empty.echoed == null, "zero budget emitted input");
    chat(module, "你好", first);
    chat(module, "你是谁", SECOND);
    module.resetContext();
    chat(module, "你好", first);
    module.resetContext();
    Output stopped = new Output();
    stopped.stopAtEcho = module;
    check(module.generateWithChat("你好", 20, stopped, true) == 0, "cancel failed");
    check(first.equals(stopped.echoed) && stopped.generated.length() == 0,
        "cancel did not precede prefill");
    chat(module, "你好", first);
    module.unload();
    chat(module, "你好", first);
    module.resetContext();
    Output plain = new Output();
    check(module.generate(FIRST, 96, plain, true) == 0, "plain generate failed");
    check(FIRST.equals(plain.echoed), "plain generate must not inject a template");
  }

  public static void main(String[] args) {
    check(args.length == 2, "EARLY.pte TOKENIZER.json");
    for (String system : new String[] {null, "", "请简洁回答。"}) {
      String first = system == null || system.isEmpty() ? FIRST
          : "<|im_start|>system\n" + system + "<|im_end|>\n" + FIRST;
      for (boolean counts : new boolean[] {false, true}) {
        try (LlmModule module = counts
            ? new LlmModule(4, args[0], args[1], system, "lincal", 0f, null, 0, 0)
            : new LlmModule(4, args[0], args[1], system, "lincal", 0f, null)) {
          checkConversation(module, first);
        }
      }
    }
    for (int mode : new int[] {0, 1}) {
      try (LlmModule module = new LlmModule(LlmModuleConfig.create()
          .modulePath(args[0]).tokenizerPath(args[1]).modelType(4)
          .qnnDecoderModel("lincal").qnnEvalMode(mode).temperature(0f).build())) {
        checkConversation(module, FIRST);
      }
    }
    System.out.println("PASS: null/empty/explicit system, KV/hybrid ChatML, reset/reload/cancel, plain input");
    System.exit(0);
  }
}
