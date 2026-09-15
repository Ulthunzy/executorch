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

/** Runs with app_process against a Lincal KV PTE; see README.md in this directory. */
public final class QnnLlmSmokeTest {
  private static final String PROMPT =
      "<|im_start|>user\n你好，你是谁？请用三句话介绍自己。<|im_end|>\n"
          + "<|im_start|>assistant\n";
  private static final String NEXT_PROMPT =
      "<|im_start|>user\n1+1是多少？<|im_end|>\n<|im_start|>assistant\n";

  private static final class Output implements LlmCallback {
    final StringBuilder text = new StringBuilder();
    String stats;
    int statsCalls;
    LlmModule stopModule;
    boolean expectError;
    int errorCode;
    String errorMessage;
    int stopDelayMs;
    Thread stopper;

    @Override
    public void onResult(String token) {
      text.append(token);
      if (stopModule != null) {
        // Exercise stop from another thread while generate holds the Kotlin lock.
        LlmModule target = stopModule;
        stopper = new Thread(() -> {
          try {
            if (stopDelayMs > 0) Thread.sleep(stopDelayMs);
          } catch (InterruptedException e) {
            throw new AssertionError(e);
          }
          target.stop();
        });
        stopper.start();
        stopModule = null;
        if (stopDelayMs > 0) return;
        try {
          stopper.join(5000);
        } catch (InterruptedException e) {
          throw new AssertionError(e);
        }
        check(!stopper.isAlive(), "stop must not wait on the generation lock");
      }
    }

    @Override
    public void onStats(String value) {
      stats = value;
      statsCalls++;
    }

    @Override
    public void onError(int code, String message) {
      if (!expectError) throw new AssertionError(code + ": " + message);
      errorCode = code;
      errorMessage = message;
    }

    int tokens() throws Exception {
      check(statsCalls == 1, "expected exactly one stats callback");
      return new JSONObject(stats).getInt("generated_tokens");
    }
  }

  private static void check(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }

  private static LlmGenerationConfig config(int maxTokens, boolean ignoreEos) {
    return LlmGenerationConfig.create()
        .echo(false)
        .seqLen(1024)
        .maxNewTokens(maxTokens)
        .ignoreEos(ignoreEos)
        .build();
  }

  private static Output generate(LlmModule module, int maxTokens, boolean ignoreEos)
      throws Exception {
    return generate(module, PROMPT, maxTokens, ignoreEos);
  }

  private static Output generate(LlmModule module, String prompt, int maxTokens, boolean ignoreEos)
      throws Exception {
    Output result = new Output();
    check(module.generate(prompt, config(maxTokens, ignoreEos), result) == 0,
        "generation failed");
    check(!result.text.toString().contains("<|im_start|>user"), "echo=false was ignored");
    if (ignoreEos) check(result.tokens() == maxTokens, "maxNewTokens was not respected");
    return result;
  }

  private static void expectFailure(LlmModule module, String prompt,
      LlmGenerationConfig generation, Output result, int expectedCode) {
    result.expectError = true;
    check(module.generate(prompt, generation, result) == expectedCode,
        "unexpected generation status");
    check(result.errorCode == expectedCode, "unexpected native error: " + result.errorMessage);
  }

  private static void checkContinuation(LlmModule module, int firstTokens, boolean ignoreEos)
      throws Exception {
    module.resetContext();
    String first = generate(module, firstTokens, ignoreEos).text.toString();
    String continued = generate(module, NEXT_PROMPT, 16, true).text.toString();
    module.resetContext();
    String replayed = generate(module, PROMPT + first + NEXT_PROMPT, 16, true).text.toString();
    check(continued.equals(replayed), "multi-turn continuation dropped the final token");
  }

  private static void checkEosContinuation(LlmModule module) throws Exception {
    module.resetContext();
    Output next = generate(module, NEXT_PROMPT, 1, true);
    int nextPromptTokens = new JSONObject(next.stats).getInt("prompt_tokens");
    module.resetContext();
    Output eos = generate(module, 256, false);
    check(eos.text.toString().endsWith("<|im_end|>"), "expected terminal EOS");
    JSONObject stats = new JSONObject(eos.stats);
    int occupied = stats.getInt("prompt_tokens") + stats.getInt("generated_tokens");
    // Text round-tripping can change BPE segmentation. Use exact token budgets
    // to check that the final EOS is retained, including when validation fails.
    expectFailure(module, NEXT_PROMPT, LlmGenerationConfig.create().echo(false)
        .seqLen(occupied + nextPromptTokens).maxNewTokens(1).build(), new Output(), 0x12);
    Output continued = new Output();
    module.generate(NEXT_PROMPT, LlmGenerationConfig.create().echo(false)
        .seqLen(occupied + nextPromptTokens + 1).maxNewTokens(1).build(), continued);
    check(continued.tokens() == 1, "EOS continuation did not respect the exact remaining budget");
  }

  private static String repeatedToken(int count) {
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < count; ++i) text.append("<|im_start|>");
    return text.toString();
  }

  private static void checkPartialPrefill(LlmModule module, int historyTokens) throws Exception {
    module.resetContext();
    String history = repeatedToken(historyTokens) + PROMPT;
    generate(module, history, 1, true);
    Output partial = new Output();
    partial.stopModule = module;
    partial.stopDelayMs = 150;
    module.generate(repeatedToken(400), LlmGenerationConfig.create().echo(true)
        .maxNewTokens(16).build(), partial);
    partial.stopper.join(5000);
    check(!partial.stopper.isAlive(), "prefill stopper did not finish");
    check(partial.tokens() == 0, "cancelled prefill generated a token");
    int committed = new JSONObject(partial.stats).getInt("prompt_tokens");
    check(committed > 0 && committed < 400, "expected partial AR-1 progress: " + committed);
    String continued = generate(module, NEXT_PROMPT, 16, true).text.toString();
    module.resetContext();
    generate(module, history, 1, true);
    String baseline = generate(module, repeatedToken(committed) + NEXT_PROMPT, 16, true)
        .text.toString();
    check(continued.equals(baseline),
        "cancelled prefill changed committed history, pending token or SWA position");
    System.out.println("PASS: partial prefill kept " + committed
        + " new tokens after " + historyTokens + " history-padding tokens");
  }

  public static void main(String[] args) throws Exception {
    check(args.length == 2, "Usage: QnnLlmSmokeTest MODEL.pte TOKENIZER.json");
    LlmModuleConfig moduleConfig =
        LlmModuleConfig.create()
            .modulePath(args[0])
            .tokenizerPath(args[1])
            .modelType(LlmModule.MODEL_TYPE_QNN_LLAMA)
            .qnnDecoderModel("lincal")
            .qnnEvalMode(LlmModuleConfig.QNN_EVAL_MODE_KV)
            .temperature(0.0f)
            .qnnRepetitionPenalty(1.0f)
            .build();
    try (LlmModule module = new LlmModule(moduleConfig)) {
      module.load();
      Output empty = generate(module, 0, true);
      check(empty.text.length() == 0, "maxNewTokens=0 emitted output");
      generate(module, 1, true);
      module.resetContext();
      String first = generate(module, 16, true).text.toString();
      module.resetContext();
      String second = generate(module, 16, true).text.toString();
      check(first.equals(second), "reset changed greedy output");
      check(first.contains("<think>"), "thinking should remain enabled");
      module.resetContext();
      generate(module, 530, true);
      module.resetContext();
      check(first.equals(generate(module, 16, true).text.toString()),
          "reset failed after crossing the SWA window");
      module.resetContext();
      Output eos = generate(module, 256, false);
      check(eos.tokens() < 256, "short identity prompt did not stop at EOS");
      check(eos.text.toString().contains("<|im_end|>"), "expected Lincal EOS");
      module.resetContext();
      Output stopped = new Output();
      stopped.stopModule = module;
      module.generate(PROMPT, config(256, true), stopped);
      check(stopped.tokens() == 1, "stop did not interrupt after prefill token");
      module.resetContext();
      Output stoppedPrefill = new Output();
      stoppedPrefill.stopModule = module;
      module.generate(PROMPT,
          LlmGenerationConfig.create().echo(true).maxNewTokens(256).build(), stoppedPrefill);
      check(stoppedPrefill.tokens() == 0, "stop during prompt echo did not cancel prefill");
      check(first.equals(generate(module, 16, true).text.toString()),
          "cancelled prefill left a partial cache");
      checkContinuation(module, 1, true);
      checkEosContinuation(module);

      module.resetContext();
      generate(module, 16, true);
      String next = generate(module, NEXT_PROMPT, 16, true).text.toString();
      module.resetContext();
      generate(module, 16, true);
      Output cancelSecondTurn = new Output();
      cancelSecondTurn.stopModule = module;
      module.generate(NEXT_PROMPT, LlmGenerationConfig.create().echo(true)
          .maxNewTokens(16).build(), cancelSecondTurn);
      check(cancelSecondTurn.tokens() == 0, "cancelled prompt emitted output");
      generate(module, 0, true);
      check(next.equals(generate(module, NEXT_PROMPT, 16, true).text.toString()),
          "cancellation/no-op erased previous history or its pending token");

      checkPartialPrefill(module, 0);
      checkPartialPrefill(module, 520);

      module.resetContext();
      Output appendedEos = new Output();
      module.generate(PROMPT, LlmGenerationConfig.create().echo(false).maxNewTokens(8)
          .ignoreEos(true).numEos(2).build(), appendedEos);
      module.resetContext();
      Output explicitEos = generate(module, PROMPT + "<|im_end|><|im_end|>", 8, true);
      check(appendedEos.text.toString().equals(explicitEos.text.toString()),
          "numEos did not append the correct EOS IDs");
      check(new JSONObject(appendedEos.stats).getInt("prompt_tokens") ==
          new JSONObject(explicitEos.stats).getInt("prompt_tokens"), "wrong EOS token count");
      module.resetContext();
      expectFailure(module, PROMPT, LlmGenerationConfig.create().numBos(1).build(),
          new Output(), 0x12);
      check(first.equals(generate(module, 16, true).text.toString()),
          "invalid BOS request changed the context");
      module.resetContext();
      Output sampled = new Output();
      module.generate(PROMPT,
          LlmGenerationConfig.create().echo(false).maxNewTokens(8)
              .temperature(0.6f).ignoreEos(true).warming(true).build(), sampled);
      check(sampled.tokens() == 8, "temperature/warming generation failed");
      module.resetContext();
      Output legacy = new Output();
      module.generate(PROMPT, 32, legacy, false, 0.0f, 0, 0);
      check(legacy.tokens() > 0, "legacy generate overload failed");
    }
    try (LlmModule shared = new LlmModule(LlmModuleConfig.create()
        .modulePath(args[0]).tokenizerPath(args[1])
        .modelType(LlmModule.MODEL_TYPE_QNN_LLAMA).qnnDecoderModel("lincal")
        .qnnEvalMode(LlmModuleConfig.QNN_EVAL_MODE_KV).qnnSharedBuffer(true)
        .temperature(0.0f).build())) {
      generate(shared, 8, true);
    }
    System.out.println("PASS: QNN JNI load, limits, EOS/counts, multi-turn, cancellation, reset/SWA, temperature, legacy API");
    System.exit(0);
  }
}
