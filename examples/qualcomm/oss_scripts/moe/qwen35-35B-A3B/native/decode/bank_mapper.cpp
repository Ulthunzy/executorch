#include "bank_mapper.h"
#include "HAP_mem.h"
#include <cstdlib>

void BankMapper::work(void* opaque) {
  auto* self = static_cast<BankMapper*>(opaque);
  while (true) {
    qurt_sem_down(&self->ready);
    if (read(&self->stop))
      break;
    const auto stamp = tick();
    auto& mapping = self->windows[self->requested % 2];
    self->status = 0;
    if (mapping.fd != self->fds[self->requested]) {
      if (mapping.pointer && HAP_munmap(mapping.pointer, mapping.bytes))
        self->status = 52;
      mapping = BankMapping{};
      if (!self->status) {
        auto* p = static_cast<uint8_t*>(HAP_mmap2(
            nullptr,
            self->bytes,
            HAP_PROT_READ | HAP_PROT_WRITE,
            0,
            self->fds[self->requested],
            0));
        if (!p || p == reinterpret_cast<void*>(-1))
          self->status = 25;
        else
          mapping = {self->fds[self->requested], 0, p, unsigned(self->bytes)};
      }
    }
    self->map_ticks = tick() - stamp;
    qurt_sem_up(&self->done);
  }
}

BankMapper::BankMapper() {
  qurt_sem_init(&ready);
  qurt_sem_down(&ready);
  qurt_sem_init(&done);
  qurt_sem_down(&done);
}

int BankMapper::configure(const int* descriptors, int count) {
  if (started || count != 41 || descriptors[40] <= 0)
    return 77;
  for (int i = 0; i < 40; i++) {
    if (descriptors[i] < 0)
      return 77;
    fds[i] = descriptors[i];
  }
  bytes = descriptors[40];
  stack = memalign(128, 32768);
  if (!stack)
    return 12;
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, 32768);
  qurt_thread_attr_set_priority(&attr, 100);
  qurt_thread_attr_set_name(&attr, "moe-bank-map");
  int error = qurt_thread_create(&thread, &attr, work, this);
  if (error)
    return error;
  started = true;
  schedule(0);
  return 0;
}

int BankMapper::wait(int fd, int length, const uint8_t*& pointer, int& layer) {
  if (pending) {
    qurt_sem_down(&done);
    pending = false;
  }
  if (status)
    return status;
  layer = -1;
  for (int i = 0; i < 40; i++)
    if (fds[i] == fd) {
      layer = i;
      break;
    }
  if (layer < 0 || length != bytes)
    return 78;
  const auto& mapping = windows[layer % 2];
  if (mapping.fd != fd || !mapping.pointer)
    return 79;
  pointer = mapping.pointer;
  return 0;
}

void BankMapper::schedule(int layer) {
  requested = layer;
  pending = true;
  qurt_sem_up(&ready);
}

void BankMapper::close() {
  if (pending) {
    qurt_sem_down(&done);
    pending = false;
  }
  if (started) {
    write(&stop, 1);
    qurt_sem_up(&ready);
    int code;
    qurt_thread_join(thread, &code);
    started = false;
  }
  for (auto& mapping : windows)
    if (mapping.pointer) {
      HAP_munmap(mapping.pointer, mapping.bytes);
      mapping = BankMapping{};
    }
}

BankMapper::~BankMapper() {
  close();
  qurt_sem_destroy(&ready);
  qurt_sem_destroy(&done);
  free(stack);
}
