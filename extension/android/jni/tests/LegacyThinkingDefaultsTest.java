/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;

/** Compile against the old App AAR; run with the new AAR and the early.pte fixture. */
public final class LegacyThinkingDefaultsTest {
  private static final class Output implements LlmCallback {
    final StringBuilder text = new StringBuilder();
    public void onResult(String piece) { text.append(piece); }
    public void onStats(String stats) {}
  }

  public static void main(String[] args) {
    for (boolean counts : new boolean[] {false, true}) {
      LlmModule module = counts
          ? new LlmModule(4, args[0], args[1], "", "lincal", 0f, null, 0, 0)
          : new LlmModule(4, args[0], args[1], "", "lincal", 0f, null);
      try {
        for (int reload = 0; reload < 2; reload++) {
          if (module.load() != 0) throw new AssertionError("legacy load failed");
          Output out = new Output();
          if (module.generateWithChat("你好", 20, out, false) != 0 ||
              !out.text.toString().contains("<think></think>OK")) {
            throw new AssertionError("legacy default EOS repair failed: " + out.text);
          }
          module.unload();
        }
      } finally {
        module.resetNative();
      }
    }
    System.out.println("PASS: old AAR bytecode, both legacy constructors, protected chat and reload");
    System.exit(0);
  }
}
