/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm

/**
 * Configuration class for initializing a LlmModule.
 *
 * Use [create] method and the fluent builder pattern.
 */
class LlmModuleConfig
private constructor(
    val modulePath: String,
    val tokenizerPath: String,
    val temperature: Float,
    val dataPath: String?,
    val modelType: Int,
    val numBos: Int,
    val numEos: Int,
    val loadMode: Int,
    val qnnDecoderModel: String,
    val qnnEvalMode: Int,
    val qnnSharedBuffer: Boolean,
    val qnnRepetitionPenalty: Float,
    val qnnModelKeyHex: String,
) {
  var qnnRepeatLastN: Int = 128
    private set

  companion object {
    /** Load entire model file into a buffer (no mmap). */
    const val LOAD_MODE_FILE = 0

    /** Load model via mmap without mlock (default). Pages faulted in on demand. */
    const val LOAD_MODE_MMAP = 1

    /** Load model via mmap and pin all pages with mlock. */
    const val LOAD_MODE_MMAP_USE_MLOCK = 2

    /** Load model via mmap and attempt mlock, ignoring mlock failures. */
    const val LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS = 3

    /** Model type constant for text-only models. */
    const val MODEL_TYPE_TEXT = 1

    /** Model type constant for text-and-vision multimodal models. */
    const val MODEL_TYPE_TEXT_VISION = 2

    /** Model type constant for generic multimodal models. */
    const val MODEL_TYPE_MULTIMODAL = 2

    const val MODEL_TYPE_MEDIATEK_LLAMA = 3
    const val MODEL_TYPE_QNN_LLAMA = 4
    const val QNN_EVAL_MODE_KV = 0
    const val QNN_EVAL_MODE_HYBRID = 1

    /**
     * Creates a new Builder instance for constructing LlmModuleConfig objects.
     *
     * @return a new Builder instance with default configuration values
     */
    @JvmStatic fun create(): Builder = Builder()
  }

  /**
   * Builder class for constructing LlmModuleConfig instances with optional parameters.
   *
   * The builder provides a fluent interface for configuring model parameters and validates required
   * fields before construction.
   */
  class Builder internal constructor() {
    private var modulePath: String? = null
    private var tokenizerPath: String? = null
    private var temperature: Float = 0.8f
    private var dataPath: String? = null
    private var modelType: Int = MODEL_TYPE_TEXT
    private var numBos: Int = 0
    private var numEos: Int = 0
    private var loadMode: Int = LOAD_MODE_MMAP
    private var qnnDecoderModel: String = "llama3"
    private var qnnEvalMode: Int = QNN_EVAL_MODE_HYBRID
    private var qnnSharedBuffer: Boolean = false
    private var qnnRepetitionPenalty: Float = 1.05f
    private var qnnRepeatLastN: Int = 128
    private var qnnModelKeyHex: String = ""

    /** Sets the path to the module. */
    fun modulePath(modulePath: String): Builder = apply { this.modulePath = modulePath }

    /** Sets the path to the tokenizer. */
    fun tokenizerPath(tokenizerPath: String): Builder = apply { this.tokenizerPath = tokenizerPath }

    /** Sets the temperature for sampling generation. */
    fun temperature(temperature: Float): Builder = apply { this.temperature = temperature }

    /** Sets the path to optional additional data files. */
    fun dataPath(dataPath: String?): Builder = apply { this.dataPath = dataPath }

    /** Sets the model type (text-only or multimodal). */
    fun modelType(modelType: Int): Builder = apply { this.modelType = modelType }

    /** Sets the number of BOS tokens to prepend. */
    fun numBos(numBos: Int): Builder = apply { this.numBos = numBos }

    /** Sets the number of EOS tokens to append. */
    fun numEos(numEos: Int): Builder = apply { this.numEos = numEos }

    /** QNN decoder identifier, e.g. "lincal", "qwen3", or "llama3". */
    fun qnnDecoderModel(value: String): Builder = apply { qnnDecoderModel = value }

    /** Must match the KV or hybrid methods compiled into the QNN PTE. */
    fun qnnEvalMode(value: Int): Builder = apply { qnnEvalMode = value }

    /** Enables QNN shared I/O buffers. */
    fun qnnSharedBuffer(value: Boolean): Builder = apply { qnnSharedBuffer = value }

    /** QNN repetition penalty. Default: 1.05; 1.0 disables it. */
    fun qnnRepetitionPenalty(value: Float): Builder = apply { qnnRepetitionPenalty = value }

    /** Number of recent prompt/generated tokens penalized by QNN. Default: 128; must be positive. */
    fun qnnRepeatLastN(value: Int): Builder = apply { qnnRepeatLastN = value }

    /** AES key for ETENC1 models; empty uses the embedded key in legacy encrypted files. */
    fun qnnModelKeyHex(value: String): Builder = apply { qnnModelKeyHex = value }

    /**
     * Sets the load mode for the model file. Defaults to [LOAD_MODE_MMAP] (mmap without mlock),
     * which avoids pinning model pages in RAM.
     *
     * @throws IllegalArgumentException if loadMode is not one of the supported constants
     */
    fun loadMode(loadMode: Int): Builder {
      require(
          loadMode == LOAD_MODE_FILE ||
              loadMode == LOAD_MODE_MMAP ||
              loadMode == LOAD_MODE_MMAP_USE_MLOCK ||
              loadMode == LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS
      ) {
        "Unknown load mode: $loadMode"
      }
      return apply { this.loadMode = loadMode }
    }

    /**
     * Constructs the LlmModuleConfig instance with validated parameters.
     *
     * @throws IllegalArgumentException if required fields are missing
     */
    fun build(): LlmModuleConfig {
      require(modulePath != null && tokenizerPath != null) {
        "Module path and tokenizer path are required"
      }
      require(temperature.isFinite() && temperature >= 0) { "Invalid temperature" }
      require(numBos in 0..127 && numEos in 0..127) { "BOS/EOS counts must be in 0..127" }
      require(qnnDecoderModel.isNotBlank()) { "QNN decoder model must not be blank" }
      require(qnnEvalMode == QNN_EVAL_MODE_KV || qnnEvalMode == QNN_EVAL_MODE_HYBRID) {
        "QNN eval mode must be KV or hybrid"
      }
      require(qnnRepetitionPenalty.isFinite() && qnnRepetitionPenalty >= 1.0f) {
        "QNN repetition penalty must be finite and >= 1.0"
      }
      require(qnnRepeatLastN > 0) { "QNN repetition window must be positive" }
      val key = qnnModelKeyHex.filterNot { it.isWhitespace() }.removePrefix("0x").removePrefix("0X")
      require(
          key.isEmpty() ||
              ((key.length == 32 || key.length == 64) &&
                  key.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' })
      ) {
        "QNN model key must contain 32 or 64 hexadecimal digits"
      }
      return LlmModuleConfig(
          modulePath!!,
          tokenizerPath!!,
          temperature,
          dataPath,
          modelType,
          numBos,
          numEos,
          loadMode,
          qnnDecoderModel,
          qnnEvalMode,
          qnnSharedBuffer,
          qnnRepetitionPenalty,
          key,
      ).also { it.qnnRepeatLastN = qnnRepeatLastN }
    }
  }
}
