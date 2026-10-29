#pragma once

#include <atomic>
#include <stdexcept>

namespace moe {
// A JNI session may be created and closed on different application threads.
class SessionLease {
 public:
  SessionLease() {
    if (active_.exchange(true))
      throw std::runtime_error("Another MoE session is active in this process");
  }
  ~SessionLease() {
    active_.store(false);
  }
  SessionLease(const SessionLease&) = delete;
  SessionLease& operator=(const SessionLease&) = delete;

 private:
  inline static std::atomic<bool> active_{false};
};
} // namespace moe
