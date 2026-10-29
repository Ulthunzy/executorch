#include "expert_readers.h"
#include "../../../../custom_op/lowbit/expert_stream.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

void ExpertReaders::perform(ExpertRead& request, void* buffer) {
  // 每个 worker 有自己的对齐 staging，随后只复制有效 payload，去掉对齐填充。
  if (request.raw_staging)
    buffer = request.raw_staging;
  auto begin = Clock::now();
  const uint8_t* payload = request.cached;
  if (!payload) {
    ssize_t got;
    do {
      got = pread(request.fd, buffer, request.read_bytes, request.file_offset);
    } while (got < 0 && errno == EINTR);
    request.pread_ms = elapsed(begin);
    if (got != ssize_t(request.read_bytes)) {
      request.error = got < 0 ? errno : EIO;
      if (request.completion)
        __atomic_store_n(
            request.completion, expert_stream::kFailed, __ATOMIC_RELEASE);
      return;
    }
    payload = static_cast<uint8_t*>(buffer) + request.prefix;
  }
  begin = Clock::now();
  if (request.cache_destination)
    memcpy(request.cache_destination, payload, request.payload_bytes);
  if (request.destination)
    memcpy(request.destination, payload, request.payload_bytes);
  request.copy_ms = elapsed(begin);
  if (request.completion)
    __atomic_store_n(
        request.completion, expert_stream::kReady, __ATOMIC_RELEASE);
}

void ExpertReaders::worker(int index) {
  int seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      changed.wait(lock, [&] {
        return stopping || epoch != seen;
      });
      if (stopping)
        return;
      seen = epoch;
    }
    while (true) {
      int job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] {
          return stopping || next < count || sealed;
        });
        if (stopping)
          return;
        if (next >= count)
          break;
        job = next++;
        if (active++ == 0)
          active_started = Clock::now();
      }
      perform(requests[job], staging[index]);
      const auto ended = Clock::now();
      {
        std::lock_guard<std::mutex> lock(mutex);
        completed = std::max(completed, ended);
        if (--active == 0)
          active_spans.emplace_back(active_started, completed);
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++finished;
    }
    changed.notify_all();
  }
}

ExpertReaders::ExpertReaders(int workers, size_t maximum)
    : staging(std::max(workers, 1), nullptr) {
  try {
    active_spans.reserve(1280);
    for (auto& pointer : staging)
      if (posix_memalign(&pointer, 4096, maximum))
        throw std::runtime_error("expert staging allocation");
    for (int i = 0; i < workers; i++)
      threads.emplace_back(&ExpertReaders::worker, this, i);
  } catch (...) {
    stop();
    throw;
  }
}

void ExpertReaders::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    stopping = true;
  }
  changed.notify_all();
  for (auto& thread : threads)
    if (thread.joinable())
      thread.join();
  for (auto& pointer : staging) {
    free(pointer);
    pointer = nullptr;
  }
}

ExpertReaders::~ExpertReaders() {
  stop();
}

void ExpertReaders::start(ExpertRead* jobs, int size, bool growable) {
  if (pending)
    throw std::runtime_error("expert read batch still pending");
  if (growable && threads.empty())
    throw std::runtime_error("growable reads require worker threads");
  requests = jobs;
  count = size;
  started = Clock::now();
  completed = started;
  active_spans.clear();
  pending = true;
  sealed = !growable;
  if (threads.empty()) {
    for (int i = 0; i < size; i++)
      perform(jobs[i], staging[0]);
    completed = Clock::now();
    if (size)
      active_spans.emplace_back(started, completed);
  } else {
    std::lock_guard<std::mutex> lock(mutex);
    next = 0;
    finished = 0;
    ++epoch;
    changed.notify_all();
  }
}

void ExpertReaders::extend(int size) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!pending || sealed || size < count)
      throw std::runtime_error("invalid expert read extension");
    count = size;
  }
  changed.notify_all();
}

void ExpertReaders::seal() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!pending)
      throw std::runtime_error("no expert read batch pending");
    sealed = true;
  }
  changed.notify_all();
}

void ExpertReaders::finish() {
  if (!pending)
    throw std::runtime_error("no expert read batch pending");
  seal();
  if (!threads.empty()) {
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] {
      return finished == int(threads.size());
    });
  }
  pending = false;
  for (int i = 0; i < count; i++)
    if (requests[i].error)
      throw std::runtime_error(
          "expert pread errno=" + std::to_string(requests[i].error));
}

double ExpertReaders::wall_ms() const {
  return std::chrono::duration<double, std::milli>(completed - started).count();
}

double ExpertReaders::overlap_ms(Clock::time_point begin, Clock::time_point end)
    const {
  double overlap = 0;
  for (const auto& span : active_spans)
    overlap += std::max(
        0.,
        std::chrono::duration<double, std::milli>(
            std::min(end, span.second) - std::max(begin, span.first))
            .count());
  return overlap;
}

void ExpertReaders::read(ExpertRead* jobs, int size) {
  start(jobs, size);
  finish();
}
