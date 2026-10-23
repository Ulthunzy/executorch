#include "native_core.h"

void NativeCore::await_mailbox(int word, uint32_t expected) {
  const auto start = std::chrono::steady_clock::now();
  unsigned polls = 0;
  while (__atomic_load_n(box() + word, __ATOMIC_ACQUIRE) != expected) {
    if (service_done.load(std::memory_order_acquire))
      throw std::runtime_error(
          "native service exited: " + std::to_string(service_status.load()));
    if ((++polls & 4095) == 0 &&
        std::chrono::steady_clock::now() - start > std::chrono::seconds(20))
      throw std::runtime_error("native mailbox timeout");
    asm volatile("yield");
  }
}

void NativeCore::start_service() {
  // ARM 服务线程持有长 RPC；周期性续调只更新调用生命周期，不重建 DSP
  // 工作线程。
  if (service.joinable() && !service_done.load(std::memory_order_acquire))
    return;
  if (!mailbox)
    mailbox =
        std::make_shared<Buffer>(runtime, 4096, Buffer::CachePolicy::Uncached);
  map_buffer(mailbox);
  if (!service.joinable()) {
    ++service_threads;
    service = std::thread([this] {
      std::unique_lock<std::mutex> lock(service_mutex);
      while (true) {
        service_changed.wait(lock, [this] {
          return service_pending || service_stopping;
        });
        if (service_stopping)
          break;
        service_pending = false;
        lock.unlock();
        const int status = integer_rpc_serve(handle, mailbox->fd);
        lock.lock();
        service_status.store(status);
        service_done.store(true, std::memory_order_release);
        service_changed.notify_all();
      }
    });
  }
  memset(box(), 0, 4096);
  generation = 0;
  service_calls = 0;
  service_began = std::chrono::steady_clock::now();
  service_status.store(0);
  service_done.store(false);
  {
    std::lock_guard<std::mutex> lock(service_mutex);
    service_pending = true;
    ++service_invocations;
  }
  service_changed.notify_all();
  await_mailbox(64, 1);
}

int NativeCore::stop_service() {
  if (!service.joinable())
    return 0;
  if (!service_done.load(std::memory_order_acquire)) {
    __atomic_store_n(box() + 1, 1U, __ATOMIC_RELAXED);
    __atomic_store_n(box(), ++generation, __ATOMIC_RELEASE);
  }
  std::unique_lock<std::mutex> lock(service_mutex);
  service_changed.wait(lock, [this] {
    return service_done.load(std::memory_order_acquire);
  });
  return service_status.load();
}

int NativeCore::submit(const int* descriptors) {
  // word 96 起是 14 个描述字段；word 0 的 generation 递增后 DSP
  // 才读取本次命令。
  start_service();
  ++mailbox_commands;
  for (int i = 0; i < 14; i++)
    __atomic_store_n(
        box() + 96 + i, uint32_t(descriptors[i]), __ATOMIC_RELAXED);
  __atomic_store_n(box() + 1, 0U, __ATOMIC_RELAXED);
  __atomic_store_n(box(), ++generation, __ATOMIC_RELEASE);
  await_mailbox(32, generation);
  const int result = int(__atomic_load_n(box() + 33, __ATOMIC_ACQUIRE));
  // The device limits one FastRPC invocation to 10 s. Renew the invocation
  // after eight tokens or 3 s while retaining context, workers, maps and
  // buffers.
  if (++service_calls == 320 ||
      std::chrono::steady_clock::now() - service_began >
          std::chrono::seconds(3)) {
    const int status = stop_service();
    if (!result)
      return status;
  }
  return result;
}

NativeCore::NativeCore(Runtime& r, const std::string&) : runtime(r) {
  remote_rpc_control_unsigned_module request{CDSP_DOMAIN_ID, 1};
  check(
      remote_session_control(
          DSPRPC_CONTROL_UNSIGNED_MODULE, &request, sizeof(request)),
      "unsigned_module");
  std::string uri = integer_rpc_URI;
#ifdef MOE_DECODE_LIBRARY
  // AR128/AR1 的 skel ABI 不同；同一进程分别命名，避免 DSP loader 复用旧模块。
  const std::string original = "libinteger_rpc_skel.so";
  uri.replace(
      uri.find(original), original.size(), "libinteger_decode_rpc_skel.so");
#endif
  check(
      integer_rpc_open(
          (uri + "&_dom=cdsp").c_str(), &handle),
      "integer_rpc_open");
  remote_rpc_control_latency latency{RPC_POLL_QOS, 9999};
  check(
      remote_handle64_control(
          handle, DSPRPC_CONTROL_LATENCY, &latency, sizeof(latency)),
      "native RPC polling");
  for (size_t n : {1 * 2048 * 4, 1 * 8 * 4, 1 * 8 * 4, 1 * 2048 * 4, 32})
    inbuf.push_back(std::make_shared<Buffer>(r, n));
  outbuf.push_back(std::make_shared<Buffer>(r, 1 * 2048 * 4));
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
    if (!use_mailbox)
      ++direct_invocations;
    const int mapped_status = use_mailbox
        ? submit(descriptors)
        : integer_rpc_run_mapped(handle, descriptors, 14);
    if (mapped_status)
      fprintf(stderr, "NATIVE_RESOURCE_FAILURE code=%d\n", mapped_status);
    check(mapped_status, "integer_rpc_run_mapped");
    if (read_output) {
      sync_buffer(outbuf[0], DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
      output_read = true;
    }
  } else
    check(
        integer_rpc_run(
            handle,
            (float*)input(0),
            1 * 2048,
            (int*)input(1),
            1 * 8,
            (float*)input(2),
            1 * 8,
            (float*)input(3),
            1 * 2048,
            (int*)input(4),
            8,
            (float*)output(0),
            1 * 2048),
        "integer_rpc_run");
  if (getenv("MOE_PROFILE")) {
    check(stop_service(), "stop native service");
    float p[96];
    check(integer_rpc_profile(handle, p, 96), "integer_rpc_profile");
    for (int i = 0; i < 12; i++) {
      printf("PROFILE %d", i);
      for (int j = 0; j < 8; j++)
        printf(" %.6f", p[i * 8 + j]);
      printf("\n");
    }
  }
}

void NativeCore::retire_inputs() {
  // 先停 mailbox 再解除旧输入映射，输出 buffer 仍可被后续阶段继续引用。
  check(stop_service(), "stop native service");
  std::vector<int> fds;
  for (int i = 0; i < 4; i++)
    fds.push_back(inbuf[i]->fd);
  if (residual)
    fds.push_back(residual->fd);
  std::sort(fds.begin(), fds.end());
  fds.erase(std::unique(fds.begin(), fds.end()), fds.end());
  if (std::find(fds.begin(), fds.end(), outbuf[0]->fd) != fds.end())
    throw std::runtime_error("retire aliases live native output");
  check(
      integer_rpc_release_fds(handle, fds.data(), fds.size()),
      "integer_rpc_release_fds");
  for (int fd : fds) {
    auto owned = std::find(owned_mappings.begin(), owned_mappings.end(), fd);
    if (owned != owned_mappings.end()) {
      auto buffer = io_mappings.at(fd);
      check(
          fastrpc_munmap(CDSP_DOMAIN_ID, fd, buffer->data, buffer->size),
          "retired native fastrpc_munmap");
      owned_mappings.erase(owned);
    }
    io_mappings.erase(fd);
  }
}

void NativeCore::release_io() {
  check(stop_service(), "stop native service");
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
  const int service_error = stop_service();
  if (service.joinable()) {
    {
      std::lock_guard<std::mutex> lock(service_mutex);
      service_stopping = true;
    }
    service_changed.notify_all();
    service.join();
  }
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
  check(service_error, "native service");
}

NativeCore::~NativeCore() {
  if (handle)
    try {
      close();
    } catch (const std::exception& error) {
      fprintf(stderr, "native cleanup failed: %s\n", error.what());
    }
}
