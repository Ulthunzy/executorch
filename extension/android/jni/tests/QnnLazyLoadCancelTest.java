/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONObject;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;

/** Compile against the old AAR. Requires a real PTE whose cold QNN load exceeds 300 ms. */
public final class QnnLazyLoadCancelTest {
  private static final class Output implements LlmCallback {
    final StringBuilder text = new StringBuilder();
    String stats;
    public void onResult(String token) { text.append(token); }
    public void onStats(String value) { stats = value; }
  }

  private static void check(boolean value, String message) {
    if (!value) throw new AssertionError(message);
  }

  private static void cancelAndResume(LlmModule module) throws Exception {
    Output cancelled = new Output();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    CountDownLatch started = new CountDownLatch(1);
    Thread worker = new Thread(() -> {
      started.countDown();
      try {
        check(module.generateWithChat("你好，你是谁？", 16, cancelled, false) == 0,
            "cancelled call failed");
      } catch (Throwable error) {
        failure.set(error);
      }
    });
    worker.start();
    started.await();
    Thread.sleep(300);
    check(worker.isAlive(), "fixture finished before cancellation; use a real cold-load PTE");
    module.stop();
    worker.join(30000);
    check(!worker.isAlive(), "cancelled lazy load timed out");
    if (failure.get() != null) throw new AssertionError(failure.get());
    JSONObject stats = new JSONObject(cancelled.stats);
    check(stats.getInt("generated_tokens") == 0, "stop during load was discarded");
    check(stats.getInt("prompt_tokens") == 0, "cancelled load still prefills input");
    check(cancelled.text.length() == 0, "cancelled load emitted tokens");
    Output resumed = new Output();
    check(module.generateWithChat("你好", 1, resumed, true) == 0, "resume failed");
    check(new JSONObject(resumed.stats).getInt("generated_tokens") == 1,
        "stop flag leaked into next call");
    check(resumed.text.toString().startsWith("<|im_start|>system\n"),
        "cancelled load incorrectly consumed the system prompt");
  }

  public static void main(String[] args) throws Exception {
    check(args.length == 2, "REAL_KV_PTE TOKENIZER");
    LlmModule module = new LlmModule(4, args[0], args[1],
        "You are a helpful assistant.", "lincal", 0.0f, Collections.emptyList());
    cancelAndResume(module);
    module.unload();
    cancelAndResume(module);
    module.resetNative();
    System.out.println("PASS: lazy chat load cancellation, unload/reload, immediate resume");
    System.exit(0);
  }
}
