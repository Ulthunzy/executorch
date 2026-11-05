#include "native_core.h"
#include "../../../../custom_op/lowbit/expert_stream.h"
#include "../../../../custom_op/lowbit/prefill/vtcm_layout.h"

NativeCore::NativeCore(Runtime& r, const std::string&, int tokens)
    : runtime(r) {
  remote_rpc_control_unsigned_module request{CDSP_DOMAIN_ID, 1};
  check(
      remote_session_control(
          DSPRPC_CONTROL_UNSIGNED_MODULE, &request, sizeof(request)),
      "unsigned_module");
  const char* package = getenv("SHARED_OP_PACKAGE");
  const std::string uri = std::string("file:///") +
      (package ? package : "libet_qnnop.so") +
      "?et_prefill_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp";
  check(
      et_prefill_rpc_open(uri.c_str(), &handle),
      "et_prefill_rpc_open");
  const int configured = et_prefill_rpc_configure(handle, tokens);
  if (configured) {
    float profile[80]{};
    const int queried = et_prefill_rpc_profile(handle, profile, 80);
    fprintf(
        stderr,
        "NATIVE_VTCM_CONFIG_FAILED tokens=%d requested_bytes=%u total_mib=%.3f available_mib=%.3f hardware_threads=%.0f query_status=%d configure_status=%d\n",
        tokens,
        unsigned(tokens > 128 ? moe_prefill_vtcm::kGroupedBytes : moe_prefill_vtcm::kAr128Bytes),
        profile[76], profile[77], profile[75], queried, configured);
    et_prefill_rpc_close(handle);
    handle = 0;
    check(configured, "et_prefill_rpc_configure");
  }
  remote_rpc_control_latency latency{RPC_POLL_QOS, 9999};
  check(
      remote_handle64_control(
          handle, DSPRPC_CONTROL_LATENCY, &latency, sizeof(latency)),
      "native RPC polling");
  for (size_t n :
       {tokens * 2048 * 4,
        tokens * 8 * 4,
        tokens * 8 * 4,
        tokens * 2048 * 4,
        32})
    inbuf.push_back(std::make_shared<Buffer>(r, n));
  outbuf.push_back(std::make_shared<Buffer>(r, tokens * 2048 * 4));
  if (mapped)
    sync_buffer(outbuf[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
}

int NativeCore::named(bool input, const char* name) {
  if (!input)
    return 0;
  std::vector<std::string> names = {
      "hidden", "ids", "scores", "shared", "bank_descriptor"};
  for (int i = 0; i < 5; i++)
    if (name == names[i])
      return i;
  throw std::runtime_error("native input");
}

void NativeCore::sync_buffer(
    const std::shared_ptr<Buffer>& buffer,
    uint64_t flags) {
  dma_buf_sync sync{flags};
  if (ioctl(buffer->fd, DMA_BUF_IOCTL_SYNC, &sync))
    throw std::runtime_error("native DMA_BUF_IOCTL_SYNC");
}

void NativeCore::publish_inputs() {
  if (mapped)
    for (int i = 0; i < 4; i++)
      sync_buffer(inbuf[i], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
}

void NativeCore::map_buffer(
    const std::shared_ptr<Buffer>& buffer,
    fastrpc_map_flags flags) {
  if (io_mappings.count(buffer->fd))
    return;
  const int status = fastrpc_mmap(
      CDSP_DOMAIN_ID, buffer->fd, buffer->data, 0, buffer->size, flags);
  if (status != AEE_EALREADY || flags == FASTRPC_MAP_FD)
    check(status, "native fastrpc_mmap");
  if (!status)
    owned_mappings.push_back(buffer->fd);
  io_mappings[buffer->fd] = buffer;
  if (flags == FASTRPC_MAP_FD)
    printf(
        "NATIVE_BANK_PREMAP fd=%d bytes=%zu before_qnn_restore=1\n",
        buffer->fd,
        buffer->size);
}

void NativeCore::end_output_read() {
  if (output_read) {
    sync_buffer(outbuf[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
    output_read = false;
  }
}

void NativeCore::execute(bool read_output) {
  const int tokens =
      expert_stream::token_count(static_cast<const int*>(input(4))[3]);
  if (mapped) {
    if (output_read) {
      sync_buffer(outbuf[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
      output_read = false;
    }
    // FD、bank 描述和分配容量分开传递，映射不随有效 token 数改变。
    int descriptors[20];
    for (int i = 0; i < 4; i++) {
      map_buffer(inbuf[i]);
      descriptors[i] = inbuf[i]->fd;
      descriptors[14 + i] = int(inbuf[i]->size);
    }
    map_buffer(outbuf[0]);
    descriptors[4] = outbuf[0]->fd;
    descriptors[18] = int(outbuf[0]->size);
    descriptors[5] = -1;
    descriptors[19] = 0;
    if (fuse_residual && residual) {
      map_buffer(residual);
      descriptors[5] = residual->fd;
      descriptors[19] = int(residual->size);
    }
    memcpy(descriptors + 6, input(4), 32);
    const int status = et_prefill_rpc_run_mapped(handle, descriptors, 20);
    if (status == 25)
      fprintf(
          stderr,
          "NATIVE_BANK_MAP_FAILED fd=%d bytes=%d generation=%d tokens=%d layer=%d\n",
          descriptors[6],
          descriptors[7],
          descriptors[8],
          tokens,
          descriptors[13]);
    check(status, "et_prefill_rpc_run_mapped");
    if (read_output) {
      sync_buffer(outbuf[0], DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
      output_read = true;
    }
  } else
    check(
        et_prefill_rpc_run(
            handle,
            (float*)input(0),
            tokens * 2048,
            (int*)input(1),
            tokens * 8,
            (float*)input(2),
            tokens * 8,
            (float*)input(3),
            tokens * 2048,
            (int*)input(4),
            8,
            (float*)output(0),
            tokens * 2048),
        "et_prefill_rpc_run");
  if (getenv("MOE_PROFILE")) {
    float p[80];
    check(et_prefill_rpc_profile(handle, p, 80), "et_prefill_rpc_profile");
    for (int i = 0; i < 10; i++) {
      printf("PROFILE %d", i);
      for (int j = 0; j < 8; j++)
        printf(" %.6f", p[i * 8 + j]);
      printf("\n");
    }
  }
}

void NativeCore::release_io() {
  end_output_read();
  check(et_prefill_rpc_release_io(handle), "et_prefill_rpc_release_io");
  for (int fd : owned_mappings) {
    auto& buffer = io_mappings.at(fd);
    check(
        fastrpc_munmap(CDSP_DOMAIN_ID, fd, buffer->data, buffer->size),
        "native window fastrpc_munmap");
  }
  owned_mappings.clear();
  io_mappings.clear();
}

void NativeCore::close() {
  if (!handle)
    return;
  end_output_read();
  int profile_error = 0;
  if (getenv("MOE_PROFILE_TOTAL") &&
      std::string(getenv("MOE_PROFILE_TOTAL")) == "1") {
    float values[88];
    profile_error = et_prefill_rpc_profile(handle, values, 88);
    if (!profile_error)
      printf(
          "MOE_NATIVE_PROFILE phase=prefill enabled=%d calls=%.0f hmx_busy_ms=%.6f hmx_loop_ms=%.6f hmx_io_blocked_ms=%.6f hvx_io_wait_worker_sum_ms=%.6f acquire_ms=%.6f launch_ms=%.6f join_ms=%.6f release_ms=%.6f rpc_body_ms=%.6f dsp_mhz=%.0f hmx_mhz=%.0f hvx_units=%.0f\n",
          int(values[80]), values[81], values[72], values[74], values[78], values[79],
          values[64], values[65], values[66], values[67], values[68],
          values[69], values[70], values[71]);
  }
  const int close_status = et_prefill_rpc_close(handle);
  handle = 0;
  int unmap_status = 0;
  for (int fd : owned_mappings) {
    auto& buffer = io_mappings.at(fd);
    const int status =
        fastrpc_munmap(CDSP_DOMAIN_ID, fd, buffer->data, buffer->size);
    if (status) {
      fprintf(stderr, "native unmap failed fd=%d status=%d\n", fd, status);
      if (!unmap_status)
        unmap_status = status;
    }
  }
  owned_mappings.clear();
  io_mappings.clear();
  check(close_status, "et_prefill_rpc_close");
  check(unmap_status, "native fastrpc_munmap");
  check(profile_error, "et_prefill_rpc_profile totals");
}

NativeCore::~NativeCore() {
  if (handle)
    try {
      close();
    } catch (const std::exception& error) {
      fprintf(stderr, "native cleanup failed: %s\n", error.what());
    }
}
