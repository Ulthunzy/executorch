#include "prefetch_queue.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

namespace moe::decode {

void PrefetchQueue::work(int index) {
  while (true) {
    std::shared_ptr<PrefetchTask> task;
    {
      std::unique_lock<std::mutex> lock(mutex);
      changed.wait(lock, [&] {
        return stopping ||
            (!jobs.empty() &&
             (!paused ||
              std::any_of(jobs.begin(), jobs.end(), [](const auto& job) {
                return job->required;
              })));
      });
      if (stopping && jobs.empty())
        return;
      auto selected =
          std::find_if(jobs.begin(), jobs.end(), [](const auto& job) {
            return job->required;
          });
      if (selected == jobs.end())
        selected = jobs.begin();
      task = *selected;
      jobs.erase(selected);
    }
    auto& r = task->request;
    task->started = Clock::now();
    ssize_t got;
    do {
      got = pread(r.fd, staging[index], r.read_bytes, r.file_offset);
    } while (got < 0 && errno == EINTR);
    r.pread_ms = elapsed(task->started);
    if (got != ssize_t(r.read_bytes))
      r.error = got < 0 ? errno : EIO;
    else {
      auto copied = Clock::now();
      memcpy(
          r.cache_destination,
          static_cast<uint8_t*>(staging[index]) + r.prefix,
          r.payload_bytes);
      r.copy_ms = elapsed(copied);
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      task->completed = Clock::now();
      task->done.store(true, std::memory_order_release);
    }
    changed.notify_all();
  }
}

PrefetchQueue::PrefetchQueue(size_t maximum) {
  try {
    for (auto& pointer : staging)
      if (posix_memalign(&pointer, 4096, maximum))
        throw std::runtime_error("prefetch staging");
    for (int i = 0; i < 2; i++)
      workers.emplace_back([this, i] {
        work(i);
      });
  } catch (...) {
    stop();
    throw;
  }
}

void PrefetchQueue::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    stopping = true;
    paused = false;
  }
  changed.notify_all();
  for (auto& thread : workers)
    if (thread.joinable())
      thread.join();
  for (auto& pointer : staging) {
    free(pointer);
    pointer = nullptr;
  }
}

PrefetchQueue::~PrefetchQueue() {
  stop();
}

void PrefetchQueue::submit(const std::shared_ptr<PrefetchTask>& task) {
  std::lock_guard<std::mutex> lock(mutex);
  task->submitted = Clock::now();
  jobs.push_back(task);
  changed.notify_all();
}

void PrefetchQueue::pause(bool value) {
  std::lock_guard<std::mutex> lock(mutex);
  paused = value;
  changed.notify_all();
}

void PrefetchQueue::wait(const std::shared_ptr<PrefetchTask>& task) {
  std::unique_lock<std::mutex> lock(mutex);
  task->required = true;
  auto found = std::find(jobs.begin(), jobs.end(), task);
  if (found != jobs.end()) {
    jobs.erase(found);
    jobs.push_front(task);
    changed.notify_all();
  }
  changed.wait(lock, [&] {
    return task->done.load(std::memory_order_acquire);
  });
  if (task->request.error)
    throw std::runtime_error(
        "prefetch read errno=" + std::to_string(task->request.error));
}

} // namespace moe::decode
