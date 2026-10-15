/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/executor/method_meta.h>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#if defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
#include <sys/resource.h>
#endif
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

// Developer-only memory accounting logs.
// Override with -DET_QNN_RUNNER_DEV_MEM_LOG=1/0 when needed.
#ifndef ET_QNN_RUNNER_DEV_MEM_LOG
#define ET_QNN_RUNNER_DEV_MEM_LOG 0
#endif

// Template struct to hold tensor data and tensor

// TODO: Refactor these struct to use TensorPtr
// see https://docs.pytorch.org/executorch/stable/extension-tensor.html

// TensorStruct whose dtype known in compile time
template <typename T>
struct TensorStruct {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  std::shared_ptr<std::vector<T>> buffer;
  T* data;
  // data size in bytes
  size_t size;
};

namespace example {

inline bool has_last_token_index_input(
    const executorch::runtime::MethodMeta& metadata) {
  if (metadata.num_inputs() == 0) {
    return false;
  }
  auto last = metadata.input_tensor_meta(metadata.num_inputs() - 1);
  return last.ok() &&
      last->scalar_type() == executorch::aten::ScalarType::Int &&
      last->sizes().size() == 1 && last->sizes()[0] == 1;
}

inline size_t kv_cache_input_start(
    const executorch::runtime::MethodMeta& metadata,
    size_t num_layers) {
  // Generation-only prefill can append a row index after the cache inputs.
  const size_t end =
      metadata.num_inputs() - (has_last_token_index_input(metadata) ? 1 : 0);
  ET_CHECK_MSG(end >= 2 * num_layers, "Invalid decoder cache input layout");
  return end - 2 * num_layers;
}

} // namespace example

namespace example::dev_mem {

struct RssSample {
  size_t current_bytes{0};
  size_t peak_bytes{0};
};

inline double bytes_to_mib(size_t bytes) {
  return static_cast<double>(bytes) / 1024.0 / 1024.0;
}

inline size_t saturating_sub(size_t lhs, size_t rhs) {
  return lhs > rhs ? lhs - rhs : 0;
}

inline size_t current_rss_bytes() {
#if ET_QNN_RUNNER_DEV_MEM_LOG && (defined(__linux__) || defined(__ANDROID__))
  std::ifstream statm("/proc/self/statm");
  long total_pages = 0;
  long resident_pages = 0;
  if (statm.good() && (statm >> total_pages >> resident_pages)) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0 && resident_pages > 0) {
      return static_cast<size_t>(resident_pages) *
          static_cast<size_t>(page_size);
    }
  }
#endif
  return 0;
}

inline size_t peak_rss_bytes() {
#if ET_QNN_RUNNER_DEV_MEM_LOG && \
    (defined(__linux__) || defined(__ANDROID__) || defined(__unix__))
  struct rusage r_usage;
  if (getrusage(RUSAGE_SELF, &r_usage) == 0) {
    return static_cast<size_t>(r_usage.ru_maxrss) * 1024;
  }
#endif
  return 0;
}

inline RssSample sample_rss() {
  return RssSample{current_rss_bytes(), peak_rss_bytes()};
}

inline void log_rss_delta(
    const char* tag,
    const RssSample& before,
    const RssSample& after) {
#if ET_QNN_RUNNER_DEV_MEM_LOG
  ET_LOG(
      Info,
      "[DEV_MEM] %s: current_rss=%.2f MiB -> %.2f MiB (+%.2f MiB), peak_rss=%.2f MiB -> %.2f MiB (+%.2f MiB)",
      tag,
      bytes_to_mib(before.current_bytes),
      bytes_to_mib(after.current_bytes),
      bytes_to_mib(saturating_sub(after.current_bytes, before.current_bytes)),
      bytes_to_mib(before.peak_bytes),
      bytes_to_mib(after.peak_bytes),
      bytes_to_mib(saturating_sub(after.peak_bytes, before.peak_bytes)));
#else
  (void)tag;
  (void)before;
  (void)after;
#endif
}

struct ProcStatusMem {
  size_t vm_rss_kb{0};
  size_t vm_hwm_kb{0};
  size_t rss_anon_kb{0};
  size_t rss_file_kb{0};
  size_t rss_shmem_kb{0};
  size_t vm_data_kb{0};
  size_t vm_swap_kb{0};
};

inline bool parse_kb_line(
    const std::string& line,
    const char* prefix,
    size_t* out_kb) {
  if (!out_kb) {
    return false;
  }
  const std::string p(prefix);
  if (line.rfind(p, 0) != 0) {
    return false;
  }
  const size_t pos = line.find_first_of("0123456789");
  if (pos == std::string::npos) {
    return false;
  }
  size_t value = 0;
  bool has_digit = false;
  for (size_t i = pos; i < line.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(line[i]);
    if (!std::isdigit(c)) {
      break;
    }
    has_digit = true;
    value = value * 10 + static_cast<size_t>(c - '0');
  }
  if (!has_digit) {
    return false;
  }
  *out_kb = value;
  return true;
}

inline ProcStatusMem read_proc_status_mem() {
  ProcStatusMem mem{};
#if ET_QNN_RUNNER_DEV_MEM_LOG && (defined(__linux__) || defined(__ANDROID__))
  std::ifstream status("/proc/self/status");
  if (!status.good()) {
    return mem;
  }
  std::string line;
  while (std::getline(status, line)) {
    if (parse_kb_line(line, "VmRSS:", &mem.vm_rss_kb) ||
        parse_kb_line(line, "VmHWM:", &mem.vm_hwm_kb) ||
        parse_kb_line(line, "RssAnon:", &mem.rss_anon_kb) ||
        parse_kb_line(line, "RssFile:", &mem.rss_file_kb) ||
        parse_kb_line(line, "RssShmem:", &mem.rss_shmem_kb) ||
        parse_kb_line(line, "VmData:", &mem.vm_data_kb) ||
        parse_kb_line(line, "VmSwap:", &mem.vm_swap_kb)) {
      continue;
    }
  }
#endif
  return mem;
}

inline void log_proc_status_mem(const char* tag) {
#if ET_QNN_RUNNER_DEV_MEM_LOG
  const ProcStatusMem mem = read_proc_status_mem();
  ET_LOG(
      Info,
      "[DEV_MEM] %s breakdown: VmRSS=%.2f MiB VmHWM=%.2f MiB RssAnon=%.2f MiB RssFile=%.2f MiB RssShmem=%.2f MiB VmData=%.2f MiB VmSwap=%.2f MiB",
      tag,
      static_cast<double>(mem.vm_rss_kb) / 1024.0,
      static_cast<double>(mem.vm_hwm_kb) / 1024.0,
      static_cast<double>(mem.rss_anon_kb) / 1024.0,
      static_cast<double>(mem.rss_file_kb) / 1024.0,
      static_cast<double>(mem.rss_shmem_kb) / 1024.0,
      static_cast<double>(mem.vm_data_kb) / 1024.0,
      static_cast<double>(mem.vm_swap_kb) / 1024.0);
#else
  (void)tag;
#endif
}

} // namespace example::dev_mem

inline size_t getDtypeSize(executorch::aten::ScalarType dtype) {
  switch (dtype) {
    case executorch::aten::ScalarType::Float:
      return sizeof(float);
    case executorch::aten::ScalarType::Double:
      return sizeof(double);
    case executorch::aten::ScalarType::Int:
      return sizeof(int32_t);
    case executorch::aten::ScalarType::Long:
      return sizeof(int64_t);
    case executorch::aten::ScalarType::Byte:
      return sizeof(uint8_t);
    case executorch::aten::ScalarType::UInt16:
      return sizeof(uint16_t);
    case executorch::aten::ScalarType::Half:
      return sizeof(executorch::aten::Half);
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported scalar type %s",
          executorch::runtime::toString(dtype));
      break;
  }
}

// TensorStruct whose dtype known in runtime, and raw file is used
struct TensorStructRaw {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  std::byte* data;
  // data size in bytes
  size_t size;
  executorch::aten::ScalarType dtype;
  size_t getElementSize() const {
    return getDtypeSize(dtype);
  }
};
