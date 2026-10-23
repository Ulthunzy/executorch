#include "expert_readers.h"

#include <sys/ioctl.h>
#include <linux/dma-buf.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

void ExpertReaders::perform(ExpertRead& request, void* buffer) {
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
      int job = next.fetch_add(1);
      if (job >= count)
        break;
      perform(requests[job], staging[index]);
      if (ready_words) {
        auto& request = requests[job];
        if (request.error)
          __atomic_store_n(
              ready_words + 1, uint32_t(request.error), __ATOMIC_RELEASE);
        if (stage_ready && !request.error) {
          const int owner = owners[job];
          const auto& source = original[owner];
          const unsigned chunk =
              (request.file_offset - source.file_offset) / chunk_bytes;
          const uint32_t done = completed_chunks[owner].fetch_or(
                                    1U << chunk, std::memory_order_acq_rel) |
              (1U << chunk);
          if (done & 1U) {
            const auto* header =
                reinterpret_cast<const uint32_t*>(source.cache_destination);
            const size_t gu_end = header[7];
            if (gu_end < 128 || gu_end > source.payload_bytes) {
              request.error = EIO;
              __atomic_store_n(
                  ready_words + 1, uint32_t(EIO), __ATOMIC_RELEASE);
            } else {
              const unsigned needed =
                  (source.prefix + gu_end + chunk_bytes - 1) / chunk_bytes;
              const uint32_t mask = (1U << needed) - 1;
              if ((done & mask) == mask &&
                  !gu_published[owner].exchange(1, std::memory_order_acq_rel)) {
                dma_buf_sync sync{DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE};
                if (ioctl(cache_fd, DMA_BUF_IOCTL_SYNC, &sync)) {
                  request.error = errno;
                  __atomic_store_n(
                      ready_words + 1,
                      uint32_t(request.error),
                      __ATOMIC_RELEASE);
                } else
                  __atomic_fetch_or(
                      ready_words + 32 * (owner + 1), 1U, __ATOMIC_RELEASE);
              }
            }
          }
        }
        if (remaining[owners[job]].fetch_sub(1, std::memory_order_acq_rel) ==
                1 &&
            !request.error) {
          // The final chunk observes all payload stores before publishing
          // this expert's ready flag.
          dma_buf_sync sync{DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE};
          if (ioctl(cache_fd, DMA_BUF_IOCTL_SYNC, &sync)) {
            request.error = errno;
            __atomic_store_n(
                ready_words + 1, uint32_t(request.error), __ATOMIC_RELEASE);
          } else if (stage_ready)
            __atomic_fetch_or(
                ready_words + 32 * (owners[job] + 1), 3U, __ATOMIC_RELEASE);
          else
            __atomic_store_n(
                ready_words + 32 * (owners[job] + 1), 1U, __ATOMIC_RELEASE);
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (++finished == int(threads.size()))
        completed = Clock::now();
    }
    changed.notify_all();
  }
}

ExpertReaders::ExpertReaders(int workers, size_t maximum)
    : staging(std::max(workers, 1), nullptr) {
  if (const char* value = getenv("EXPERT_READ_CHUNK_KIB"))
    chunk_bytes = size_t(atoi(value)) * 1024;
  if (chunk_bytes &&
      (chunk_bytes < 4096 || chunk_bytes > maximum || chunk_bytes % 4096))
    throw std::runtime_error("invalid expert read chunk size");
  chunks.reserve(128);
  owners.reserve(128);
  try {
    for (auto& pointer : staging)
      if (posix_memalign(&pointer, 4096, chunk_bytes ? chunk_bytes : maximum))
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

void ExpertReaders::start(
    ExpertRead* jobs,
    int size,
    uint32_t* readiness,
    int destination_fd) {
  // 一个实例同一时间只处理一个批次；epoch 唤醒整组 worker。
  if (pending)
    throw std::runtime_error("expert read batch still pending");
  ready_words = readiness;
  cache_fd = destination_fd;
  if (ready_words &&
      (!chunk_bytes || threads.empty() || size != 8 || cache_fd < 0))
    throw std::runtime_error(
        "ready queue requires chunk readers and eight assignments");
  original = jobs;
  original_count = size;
  requests = jobs;
  count = size;
  if (chunk_bytes) {
    chunks.clear();
    owners.clear();
    for (int job = 0; job < size; job++) {
      const auto& source = jobs[job];
      if (source.cached) {
        if (source.destination || source.cache_destination) {
          chunks.push_back(source);
          owners.push_back(job);
        }
        continue;
      }
      // Chunks cover the same aligned disk range and copy disjoint portions
      // of the payload.
      for (size_t offset = 0; offset < source.read_bytes;
           offset += chunk_bytes) {
        auto chunk = source;
        chunk.file_offset += offset;
        chunk.read_bytes = std::min(chunk_bytes, source.read_bytes - offset);
        const size_t first = std::max(offset, source.prefix),
                     last = std::min(
                         offset + chunk.read_bytes,
                         source.prefix + source.payload_bytes);
        chunk.prefix = first - offset;
        chunk.payload_bytes = last > first ? last - first : 0;
        chunk.destination = source.destination
            ? source.destination + (first - source.prefix)
            : nullptr;
        chunk.cache_destination = source.cache_destination
            ? source.cache_destination + (first - source.prefix)
            : nullptr;
        chunks.push_back(chunk);
        owners.push_back(job);
      }
    }
    requests = chunks.data();
    count = chunks.size();
  }
  if (ready_words) {
    for (auto& n : remaining)
      n = 0;
    for (auto& n : completed_chunks)
      n = 0;
    for (auto& n : gu_published)
      n = 0;
    if (stage_ready)
      for (int job = 0; job < size; job++)
        if (jobs[job].read_bytes / chunk_bytes >= 31)
          throw std::runtime_error("expert stage chunk count");
    for (int owner : owners)
      ++remaining[owner];
  }
  started = Clock::now();
  pending = true;
  if (threads.empty()) {
    for (int i = 0; i < count; i++)
      perform(requests[i], staging[0]);
    completed = Clock::now();
  } else {
    std::lock_guard<std::mutex> lock(mutex);
    next = 0;
    finished = 0;
    if (count) {
      ++epoch;
      changed.notify_all();
    } else {
      finished = threads.size();
      completed = Clock::now();
    }
  }
}

void ExpertReaders::finish() {
  if (!pending)
    throw std::runtime_error("no expert read batch pending");
  if (!threads.empty()) {
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] {
      return finished == int(threads.size());
    });
  }
  pending = false;
  if (chunk_bytes) {
    for (int i = 0; i < original_count; i++) {
      original[i].pread_ms = 0;
      original[i].copy_ms = 0;
    }
    for (int i = 0; i < count; i++) {
      original[owners[i]].pread_ms += requests[i].pread_ms;
      original[owners[i]].copy_ms += requests[i].copy_ms;
    }
  }
  for (int i = 0; i < count; i++)
    if (requests[i].error)
      throw std::runtime_error(
          "expert pread errno=" + std::to_string(requests[i].error));
}

double ExpertReaders::wall_ms() const {
  return std::chrono::duration<double, std::milli>(completed - started).count();
}

uint64_t ExpertReaders::disk_bytes() const {
  uint64_t total = 0;
  for (int i = 0; i < count; i++)
    if (!requests[i].cached)
      total += requests[i].read_bytes;
  return total;
}

int ExpertReaders::disk_calls() const {
  int total = 0;
  for (int i = 0; i < count; i++)
    total += !requests[i].cached;
  return total;
}

double ExpertReaders::overlap_ms(Clock::time_point begin, Clock::time_point end)
    const {
  return std::max(
      0.,
      std::chrono::duration<double, std::milli>(
          std::min(end, completed) - std::max(begin, started))
          .count());
}

void ExpertReaders::read(ExpertRead* jobs, int size) {
  start(jobs, size);
  finish();
}
