#include "native_core.h"

NativeCore::NativeCore(Runtime& r, const std::string&) : runtime(r) {
  remote_rpc_control_unsigned_module request{CDSP_DOMAIN_ID, 1};
  check(
      remote_session_control(
          DSPRPC_CONTROL_UNSIGNED_MODULE, &request, sizeof(request)),
      "unsigned_module");
  check(
      integer_rpc_open(
          (std::string(integer_rpc_URI) + "&_dom=cdsp").c_str(), &handle),
      "integer_rpc_open");
  remote_rpc_control_latency latency{RPC_POLL_QOS, 9999};
  check(
      remote_handle64_control(
          handle, DSPRPC_CONTROL_LATENCY, &latency, sizeof(latency)),
      "native RPC polling");
  for (size_t n :
       {128 * 2048 * 4, 128 * 8 * 4, 128 * 8 * 4, 128 * 2048 * 4, 32})
    inbuf.push_back(std::make_shared<Buffer>(r, n));
  outbuf.push_back(std::make_shared<Buffer>(r, 128 * 2048 * 4));
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

void NativeCore::map_buffer(const std::shared_ptr<Buffer>& buffer) {
  if (io_mappings.count(buffer->fd))
    return;
  const int status = fastrpc_mmap(
      CDSP_DOMAIN_ID,
      buffer->fd,
      buffer->data,
      0,
      buffer->size,
      FASTRPC_MAP_FD_DELAYED);
  if (status != AEE_EALREADY)
    check(status, "native fastrpc_mmap");
  if (!status)
    owned_mappings.push_back(buffer->fd);
  io_mappings[buffer->fd] = buffer;
}

void NativeCore::end_output_read() {
  if (output_read) {
    sync_buffer(outbuf[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
    output_read = false;
  }
}

void NativeCore::execute(bool read_output) {
  if (mapped) {
    if (output_read) {
      sync_buffer(outbuf[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
      output_read = false;
    }
    // 六个 IO fd，后接八个 bank 描述字段；与 native/prefill 的 RPC ABI 对齐。
    int descriptors[14];
    for (int i = 0; i < 4; i++) {
      map_buffer(inbuf[i]);
      descriptors[i] = inbuf[i]->fd;
    }
    map_buffer(outbuf[0]);
    descriptors[4] = outbuf[0]->fd;
    descriptors[5] = -1;
    if (fuse_residual && residual) {
      map_buffer(residual);
      descriptors[5] = residual->fd;
    }
    memcpy(descriptors + 6, input(4), 32);
    check(
        integer_rpc_run_mapped(handle, descriptors, 14),
        "integer_rpc_run_mapped");
    if (read_output) {
      sync_buffer(outbuf[0], DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
      output_read = true;
    }
  } else
    check(
        integer_rpc_run(
            handle,
            (float*)input(0),
            128 * 2048,
            (int*)input(1),
            128 * 8,
            (float*)input(2),
            128 * 8,
            (float*)input(3),
            128 * 2048,
            (int*)input(4),
            8,
            (float*)output(0),
            128 * 2048),
        "integer_rpc_run");
  if (getenv("MOE_PROFILE")) {
    float p[80];
    check(integer_rpc_profile(handle, p, 80), "integer_rpc_profile");
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
  check(integer_rpc_release_io(handle), "integer_rpc_release_io");
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
  check(integer_rpc_close(handle), "integer_rpc_close");
  handle = 0;
  for (int fd : owned_mappings) {
    auto& buffer = io_mappings.at(fd);
    check(
        fastrpc_munmap(CDSP_DOMAIN_ID, fd, buffer->data, buffer->size),
        "native fastrpc_munmap");
  }
  owned_mappings.clear();
  io_mappings.clear();
}

NativeCore::~NativeCore() {
  if (handle)
    try {
      close();
    } catch (const std::exception& error) {
      fprintf(stderr, "native cleanup failed: %s\n", error.what());
    }
}
