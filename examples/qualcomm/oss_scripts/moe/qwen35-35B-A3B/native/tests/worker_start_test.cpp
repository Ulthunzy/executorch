#include "../worker_start.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

static void check_group(const std::array<int, 9>& lock_errors) {
  WorkerStart startup(lock_errors.size());
  std::atomic<int> attempted{0}, entered{0}, unlocked{0};
  std::atomic<bool> entered_early{false};
  std::array<int, 9> status{};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < lock_errors.size(); ++worker) {
    workers.emplace_back([&, worker] {
      for (size_t delay = 0; delay < worker; ++delay)
        std::this_thread::yield();
      attempted.fetch_add(1);
      status[worker] = startup.arrive(lock_errors[worker]);
      if (!status[worker]) {
        if (attempted.load() != int(lock_errors.size()))
          entered_early.store(true);
        entered.fetch_add(1);
      }
      if (!lock_errors[worker])
        unlocked.fetch_add(1);
    });
  }
  for (auto& worker : workers)
    worker.join();

  const int failures =
      std::count_if(lock_errors.begin(), lock_errors.end(), [](int value) {
        return value != 0;
      });
  if (entered_early.load() || entered.load() != (failures ? 0 : 9) ||
      unlocked.load() != 9 - failures)
    throw std::runtime_error(
        "worker group entered computation or leaked a resource lock");
  for (int result : status) {
    if (result != status[0] || (failures && !result) || (!failures && result))
      throw std::runtime_error("workers did not observe a common lock failure");
  }
  if (failures &&
      std::find(lock_errors.begin(), lock_errors.end(), status[0]) ==
          lock_errors.end())
    throw std::runtime_error("lost the original resource-lock error");
}

int main() try {
  int groups = 0;
  for (int repeat = 0; repeat < 16; ++repeat) {
    for (int failed = -1; failed < 9; ++failed) {
      std::array<int, 9> errors{};
      if (failed >= 0)
        errors[failed] = 1000 + failed + 1;
      check_group(errors);
      ++groups;
    }
  }
  check_group({1001, 0, 1003, 0, 0, 0, 0, 0, 1009});
  check_group({1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009});
  check_group({});
  std::printf(
      "PASS: %d groups, all HVX/HMX failure positions, multiple failures and recovery\n",
      groups + 3);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "FAIL: %s\n", error.what());
  return 1;
}
