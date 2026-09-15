/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import java.util.Collections;
import org.pytorch.executorch.ExecutorchRuntimeException;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmModuleConfig;

public final class QnnEncryptedModelTest {
  private static final String KEY128 = "000102030405060708090a0b0c0d0e0f";
  private static final String KEY256 = KEY128 + "101112131415161718191a1b1c1d1e1f";

  private static void check(boolean value, String message) {
    if (!value) throw new AssertionError(message);
  }

  private static LlmModuleConfig.Builder builder(String path, String tokenizer, String key) {
    return LlmModuleConfig.create().modulePath(path).tokenizerPath(tokenizer)
        .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA).qnnDecoderModel("qwen3")
        .qnnEvalMode(LlmModuleConfig.QNN_EVAL_MODE_HYBRID).temperature(0.0f)
        .qnnModelKeyHex(key);
  }

  private static String generate(LlmModule module) {
    StringBuilder text = new StringBuilder();
    int result = module.generate("你好", LlmGenerationConfig.create().echo(false)
        .maxNewTokens(1).ignoreEos(true).build(), new LlmCallback() {
          public void onResult(String token) { text.append(token); }
          public void onStats(String stats) {}
        });
    check(result == 0 && text.length() > 0, "fixture generation failed");
    return text.toString();
  }

  private static void reject(LlmModuleConfig.Builder config, int expected) {
    try (LlmModule unused = new LlmModule(config.build())) {
      throw new AssertionError("bad container accepted");
    } catch (ExecutorchRuntimeException error) {
      check(error.getErrorCode() == expected, "wrong error: " + error);
      check(!error.getMessage().contains(KEY128), "key leaked in error");
    }
  }

  public static void main(String[] args) throws Exception {
    check(args.length == 3, "PLAIN_HYBRID_AR4_PTE TOKENIZER ENCRYPTED_FIXTURE_DIR");
    String expected;
    try (LlmModule plain = new LlmModule(builder(args[0], args[1], "").build())) {
      expected = generate(plain);
    }
    try (LlmModule legacy = new LlmModule(4, args[2] + "/legacy.pte", args[1], "",
        "qwen3", 0.0f, Collections.emptyList())) {
      check(expected.equals(generate(legacy)), "legacy decryption changed outputs");
      legacy.unload();
      check(expected.equals(generate(legacy)), "legacy unload/reload failed");
    }
    for (String bits : new String[] {"128", "256"}) {
      String key = bits.equals("128") ? KEY128 : KEY256;
      try (LlmModule encrypted = new LlmModule(builder(args[2] + "/etenc" + bits + ".pte",
          args[1], "0x " + key).build())) {
        check(expected.equals(generate(encrypted)), "ETENC" + bits + " output mismatch");
        encrypted.unload();
        check(expected.equals(generate(encrypted)), "ETENC unload/reload failed");
      }
    }
    for (String dataPath : new String[] {null, ""}) {
      for (String[] model : new String[][] {
          {args[0], ""},
          {args[2] + "/legacy.pte", ""},
          {args[2] + "/etenc128.pte", KEY128},
          {args[2] + "/etenc256.pte", KEY256}}) {
        try (LlmModule optionalData = new LlmModule(builder(model[0], args[1], model[1])
            .dataPath(dataPath).build())) {
          check(expected.equals(generate(optionalData)), "empty/null dataPath output mismatch");
          optionalData.unload();
          check(expected.equals(generate(optionalData)), "empty/null dataPath reload failed");
        }
      }
    }
    reject(builder(args[2] + "/etenc256.pte", args[1], ""), 0x12);
    reject(builder(args[2] + "/etenc256.pte", args[1], KEY128), 0x23);
    reject(builder(args[2] + "/damaged.pte", args[1], KEY256), 0x23);
    reject(builder(args[2] + "/truncated.pte", args[1], KEY256), 0x23);
    reject(builder(args[2] + "/legacy.pte", args[1], "").dataPath("external.ptd"), 0x10);
    reject(builder(args[2] + "/legacy.pte", args[1], "").dataPath(" "), 0x10);
    System.out.println("PASS: legacy and ETENC1 AES-128/256, generation, reload, optional data paths, authentication failures");
    System.exit(0);
  }
}
