/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm

/**
 * Configuration class for controlling text generation parameters in LLM operations.
 *
 * This class provides settings for text generation behavior including output formatting, generation
 * limits, and sampling parameters. Instances should be created using the [create] method and the
 * fluent builder pattern.
 */
class LlmGenerationConfig
private constructor(
    @get:JvmName("isEcho") val echo: Boolean,
    val maxNewTokens: Int,
    @get:JvmName("isWarming") val warming: Boolean,
    val seqLen: Int,
    val temperature: Float,
    val numBos: Int,
    val numEos: Int,
    @get:JvmName("isIgnoreEos") val ignoreEos: Boolean,
) {
  // Keep the constructor descriptor compatible with the shipped AAR.
  @get:JvmName("isRepairThinkEos")
  var repairThinkEos: Boolean = !ignoreEos
    private set

  @get:JvmName("isEnableThinkBudget")
  var enableThinkBudget: Boolean = !ignoreEos
    private set

  var thinkBudget: Int = 512
    private set

  internal var repairThinkEosExplicit: Boolean = false
    private set

  internal var enableThinkBudgetExplicit: Boolean = false
    private set

  companion object {
    /**
     * Creates a new Builder instance for constructing generation configurations.
     *
     * @return a new Builder with default configuration values
     */
    @JvmStatic fun create(): Builder = Builder()
  }

  /**
   * Builder class for constructing LlmGenerationConfig instances.
   *
   * Provides a fluent interface for configuring generation parameters with sensible defaults. All
   * methods return the builder instance to enable method chaining.
   */
  class Builder internal constructor() {
    private var echo: Boolean = true
    private var maxNewTokens: Int = -1
    private var warming: Boolean = false
    private var seqLen: Int = -1
    private var temperature: Float = -1.0f
    private var numBos: Int = -1
    private var numEos: Int = -1
    private var ignoreEos: Boolean = false
    private var repairThinkEos: Boolean? = null
    private var enableThinkBudget: Boolean? = null
    private var thinkBudget: Int = 512

    /** Sets whether to include the input prompt in the generated output. */
    fun echo(echo: Boolean): Builder = apply { this.echo = echo }

    /** Sets the maximum number of new tokens to generate. */
    fun maxNewTokens(maxNewTokens: Int): Builder = apply { this.maxNewTokens = maxNewTokens }

    /** Enables or disables model warming. */
    fun warming(warming: Boolean): Builder = apply { this.warming = warming }

    /** Sets the maximum sequence length for generation. */
    fun seqLen(seqLen: Int): Builder = apply { this.seqLen = seqLen }

    /** Sets the temperature for random sampling. */
    fun temperature(temperature: Float): Builder = apply { this.temperature = temperature }

    /** Sets the number of BOS tokens to prepend. */
    fun numBos(numBos: Int): Builder = apply { this.numBos = numBos }

    /** Sets the number of EOS tokens to append. */
    fun numEos(numEos: Int): Builder = apply { this.numEos = numEos }

    /** Ignores stop tokens for benchmarking; disables unspecified thinking protections. */
    fun ignoreEos(ignoreEos: Boolean): Builder = apply { this.ignoreEos = ignoreEos }

    /** QNN only: repair EOS inside thinking. Defaults to true, unless ignoreEos is enabled. */
    fun repairThinkEos(enabled: Boolean): Builder = apply { repairThinkEos = enabled }

    /** QNN only: close thinking at its token budget. Defaults to true, unless ignoreEos is enabled. */
    fun enableThinkBudget(enabled: Boolean): Builder = apply { enableThinkBudget = enabled }

    /** Maximum generated thinking tokens, excluding the opening/closing markers. */
    fun thinkBudget(tokens: Int): Builder = apply { thinkBudget = tokens }

    /** Constructs the LlmGenerationConfig instance with the configured parameters. */
    fun build(): LlmGenerationConfig {
      require(seqLen == -1 || seqLen > 0) { "seqLen must be -1 or positive" }
      require(maxNewTokens >= -1) { "maxNewTokens must be -1 or non-negative" }
      require(temperature.isFinite()) { "temperature must be finite" }
      require(numBos in -1..127 && numEos in -1..127) { "BOS/EOS counts must be in -1..127" }
      require(thinkBudget > 0) { "thinkBudget must be positive" }
      val repair = repairThinkEos ?: !ignoreEos
      val budget = enableThinkBudget ?: !ignoreEos
      require(!ignoreEos || !(repair || budget)) {
        "Thinking controls cannot be combined with ignoreEos"
      }
      return LlmGenerationConfig(
              echo,
              maxNewTokens,
              warming,
              seqLen,
              temperature,
              numBos,
              numEos,
              ignoreEos,
          )
          .also {
            it.repairThinkEos = repair
            it.enableThinkBudget = budget
            it.thinkBudget = thinkBudget
            it.repairThinkEosExplicit = repairThinkEos != null
            it.enableThinkBudgetExplicit = enableThinkBudget != null
          }
    }
  }
}
