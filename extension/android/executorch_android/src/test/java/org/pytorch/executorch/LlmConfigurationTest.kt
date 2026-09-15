/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModuleConfig

class LlmConfigurationTest {
  @Test
  fun generationDefaultsInheritModuleSettings() {
    val config = LlmGenerationConfig.create().build()
    assertEquals(-1, config.maxNewTokens)
    assertEquals(-1, config.seqLen)
    assertEquals(-1, config.numBos)
    assertEquals(-1, config.numEos)
    assertEquals(-1.0f, config.temperature, 0.0f)
    assertFalse(config.ignoreEos)
    assertTrue(config.repairThinkEos)
    assertTrue(config.enableThinkBudget)
    assertEquals(512, config.thinkBudget)
    assertFalse(config.repairThinkEosExplicit)
    assertFalse(config.enableThinkBudgetExplicit)
  }

  @Test
  fun benchmarkModeDisablesOnlyUnspecifiedProtections() {
    val builder = LlmGenerationConfig.create().ignoreEos(true)
    val config = builder.build()
    assertFalse(config.repairThinkEos)
    assertFalse(config.enableThinkBudget)
    val restored = builder.ignoreEos(false).build()
    assertTrue(restored.repairThinkEos)
    assertTrue(restored.enableThinkBudget)
    assertFalse(builder.repairThinkEos(false).enableThinkBudget(false).build().repairThinkEos)
  }

  @Test
  fun thinkingControlsAreIndependentAndPerGeneration() {
    for (repair in listOf(false, true)) {
      for (budget in listOf(false, true)) {
        val config =
            LlmGenerationConfig.create()
                .repairThinkEos(repair)
                .enableThinkBudget(budget)
                .thinkBudget(256)
                .build()
        assertEquals(repair, config.repairThinkEos)
        assertEquals(budget, config.enableThinkBudget)
        assertEquals(256, config.thinkBudget)
        assertTrue(config.repairThinkEosExplicit)
        assertTrue(config.enableThinkBudgetExplicit)
      }
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().thinkBudget(0).build()
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().repairThinkEos(true).ignoreEos(true).build()
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().enableThinkBudget(true).ignoreEos(true).build()
    }
  }

  @Test
  fun qnnLincalKvConfiguration() {
    val config =
        LlmModuleConfig.create()
            .modulePath("model.pte")
            .tokenizerPath("tokenizer.json")
            .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA)
            .qnnDecoderModel("lincal")
            .qnnEvalMode(LlmModuleConfig.QNN_EVAL_MODE_KV)
            .temperature(0.0f)
            .build()
    assertEquals(4, config.modelType)
    assertEquals("lincal", config.qnnDecoderModel)
    assertEquals(0, config.qnnEvalMode)
    assertFalse(config.qnnSharedBuffer)
    assertEquals(1.05f, config.qnnRepetitionPenalty, 0.0f)
    assertEquals(128, config.qnnRepeatLastN)
  }

  @Test
  fun qnnRepetitionWindowCanBeOverridden() {
    val builder = LlmModuleConfig.create().modulePath("model.pte").tokenizerPath("tokenizer.json")
    for (window in listOf(64, 128, 256)) {
      assertEquals(window, builder.qnnRepeatLastN(window).build().qnnRepeatLastN)
    }
    assertThrows(IllegalArgumentException::class.java) { builder.qnnRepeatLastN(-1).build() }
    assertThrows(IllegalArgumentException::class.java) { builder.qnnRepeatLastN(0).build() }
  }

  @Test
  fun qnnRepetitionPenaltyCanBeDisabledOrOverridden() {
    for (penalty in listOf(1.0f, 1.1f, 1.5f)) {
      val config =
          LlmModuleConfig.create()
              .modulePath("model.pte")
              .tokenizerPath("tokenizer.json")
              .modelType(LlmModuleConfig.MODEL_TYPE_QNN_LLAMA)
              .qnnRepetitionPenalty(penalty)
              .build()
      assertEquals(penalty, config.qnnRepetitionPenalty, 0.0f)
    }
  }

  @Test
  fun invalidGenerationValuesAreRejectedBeforeJni() {
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().temperature(Float.NaN).build()
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().maxNewTokens(-2).build()
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().seqLen(0).build()
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().numBos(128).build()
    }
    assertThrows(IllegalArgumentException::class.java) {
      LlmGenerationConfig.create().numEos(256).build()
    }
  }

  @Test
  fun invalidQnnValuesAreRejectedBeforeJni() {
    val builder = LlmModuleConfig.create().modulePath("model.pte").tokenizerPath("tokenizer.json")
    assertThrows(IllegalArgumentException::class.java) { builder.qnnEvalMode(2).build() }
    for (penalty in
        listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -1.0f, 0.9f)) {
      assertThrows(IllegalArgumentException::class.java) {
        builder.qnnEvalMode(0).qnnRepetitionPenalty(penalty).build()
      }
    }
  }

  @Test
  fun specialTokenCountsRespectTokenizerRange() {
    assertEquals(127, LlmGenerationConfig.create().numEos(127).build().numEos)
    val builder = LlmModuleConfig.create().modulePath("model.pte").tokenizerPath("tokenizer.json")
    assertThrows(IllegalArgumentException::class.java) { builder.numBos(128).build() }
    assertThrows(IllegalArgumentException::class.java) { builder.numBos(0).numEos(128).build() }
  }

  @Test
  fun encryptedModelKeysAreNormalizedWithoutChangingDefaults() {
    val builder = LlmModuleConfig.create().modulePath("model.pte").tokenizerPath("tokenizer.json")
    assertEquals("", builder.build().qnnModelKeyHex)
    assertEquals(
        "ab".repeat(16),
        builder.qnnModelKeyHex("0x" + "ab ".repeat(16)).build().qnnModelKeyHex,
    )
    assertEquals(
        "CD".repeat(32),
        builder.qnnModelKeyHex("0X" + "CD".repeat(32)).build().qnnModelKeyHex,
    )
  }

  @Test
  fun invalidEncryptedModelKeysAreRejectedWithoutEchoingTheKey() {
    val builder = LlmModuleConfig.create().modulePath("model.pte").tokenizerPath("tokenizer.json")
    for (key in listOf("ab".repeat(24), "zz".repeat(16), "a".repeat(31))) {
      val error =
          assertThrows(IllegalArgumentException::class.java) { builder.qnnModelKeyHex(key).build() }
      assertFalse(error.message!!.contains(key))
    }
  }
}
