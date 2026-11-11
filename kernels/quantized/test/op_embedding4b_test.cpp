/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/quantized/NativeFunctions.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <cstring>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;
using torch::executor::native::quantized_embedding_4bit_dtype_out;
using torch::executor::native::quantized_embedding_4bit_out;

using torch::executor::testing::TensorFactory;

TEST(OpQuantizedEmbedding4bTest, FloatGroupedEmbeddingPackingAndScales) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;
  TensorFactory<ScalarType::Int> tfi;
  const std::vector<float> scales = {
      0.1f, -0.37f, 1.234567f, 0.0f, -0.0f, 0.000031f, 2.5f, 0.13f, -1.0f};
  const int rows[] = {2, 0, 2, 1};
  for (int group_size : {32, 64, 48}) {
    const int dim = group_size * 3;
    std::vector<uint8_t> packed(3 * dim / 2);
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = static_cast<uint8_t>(((i % 16) << 4) | ((i * 7 + 3) % 16));
    }
    Tensor weight = tfb.make({3, dim / 2}, packed);
    Tensor weight_scales = tf.make({3, 3}, scales);
    std::vector<float> expected(4 * dim);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < dim; ++j) {
        const uint8_t code = packed[rows[i] * dim / 2 + j / 2];
        const int value = (j % 2 == 0 ? code >> 4 : code & 15) - 8;
        expected[i * dim + j] =
            (static_cast<float>(value) - 0.0f) *
            scales[rows[i] * 3 + j / group_size];
      }
    }
    for (const Tensor& indices :
         {tfl.make({4}, {2, 0, 2, 1}), tfi.make({4}, {2, 0, 2, 1})}) {
      Tensor out = tf.zeros({4, dim});
      quantized_embedding_4bit_out(
          weight, weight_scales, std::nullopt, -8, 7, indices, out);
      EXPECT_EQ(
          std::memcmp(
              out.const_data_ptr<float>(),
              expected.data(),
              expected.size() * sizeof(float)),
          0);
      quantized_embedding_4bit_dtype_out(
          weight,
          weight_scales,
          std::nullopt,
          -8,
          7,
          indices,
          ScalarType::Float,
          out);
      EXPECT_EQ(
          std::memcmp(
              out.const_data_ptr<float>(),
              expected.data(),
              expected.size() * sizeof(float)),
          0);
    }
  }
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbedding) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({3}, {0.5, 1.0, 1.5});
  Tensor weight_zero_points = tf.make({3}, {1, -5, 0});

  // -3,  1,  6, 7,
  //  2, -5, -4, 0,
  // -8,  3, -1, 6,

  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});

  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make(
      {3, 4}, {-2.0, 0.0, 2.5, 3.0, -12.0, 4.5, -1.5, 9.0, 7.0, 0.0, 1.0, 5.0});

  quantized_embedding_4bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);

  out = tf.zeros({3, 4});
  auto context = KernelRuntimeContext();
  torch::executor::native::quantized_embedding_4bit_out(
      context,
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);

  // Groupwise quantization. groupsize = 2
  weight_scales = tf.make({3, 2}, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
  weight_zero_points = tf.make({3, 2}, {1, -5, 0, 2, -3, -1});
  /*
  fp_weight = [-2.0,  0.0,  11.0, 12.0,
                3.0, -7.5, -12.0, -4.0,
              -12.5, 15.0,   0.0, 21.0]
  */

  out = tf.zeros({3, 4});
  expected = tf.make(
      {3, 4},
      {-2.0, 0.0, 11.0, 12.0, -12.5, 15.0, 0.0, 21.0, 3.0, -7.5, -12.0, -4.0});

  quantized_embedding_4bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbeddingInt32Indices) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tfi;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({3}, {0.5, 1.0, 1.5});
  Tensor weight_zero_points = tf.make({3}, {1, -5, 0});

  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});

  Tensor indices = tfi.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make(
      {3, 4}, {-2.0, 0.0, 2.5, 3.0, -12.0, 4.5, -1.5, 9.0, 7.0, 0.0, 1.0, 5.0});

  quantized_embedding_4bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);

  out = tf.zeros({3, 4});
  auto context = KernelRuntimeContext();
  torch::executor::native::quantized_embedding_4bit_out(
      context,
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbeddingDeath1) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({4}, {0.5, 1.0, 1.5, 3.3});
  Tensor weight_zero_points = tf.make({4}, {1, 5, 7, 5});
  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});
  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  ET_EXPECT_DEATH(
      quantized_embedding_4bit_out(
          qweight,
          weight_scales,
          weight_zero_points,
          quant_min,
          quant_max,
          indices,
          out),
      "");
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbeddingBFloat16) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::BFloat16> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({3}, {0.5, 1.0, 1.5});
  Tensor weight_zero_points = tf.make({3}, {1, -5, 0});
  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});
  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make(
      {3, 4}, {-2.0, 0.0, 2.5, 3.0, -12.0, 4.5, -1.5, 9.0, 7.0, 0.0, 1.0, 5.0});

  quantized_embedding_4bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_CLOSE(out, expected);

  // Same values through the dtype_out variant.
  out = tf.zeros({3, 4});
  quantized_embedding_4bit_dtype_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      ScalarType::BFloat16,
      out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbeddingDeath2) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({2}, {0.5, 1.0});
  Tensor weight_zero_points = tf.make({2}, {1, 5});
  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});
  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  ET_EXPECT_DEATH(
      quantized_embedding_4bit_out(
          qweight,
          weight_scales,
          weight_zero_points,
          quant_min,
          quant_max,
          indices,
          out),
      "");
}
