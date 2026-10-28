import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

STUB = r"""
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

struct NativeCore;
struct Completion {
  NativeCore* core;
  bool publish_ack = true;
  int result = 0, status = 0, loads = 0;
  bool load(std::memory_order);
};
struct NativeCore {
  uint32_t words[128]{};
  uint32_t generation = 0;
  uint64_t mailbox_commands = 0;
  std::atomic<int> service_status{0};
  Completion service_done{this};
  std::mutex service_mutex;
  std::condition_variable service_changed;
  uint32_t* box() { return words; }
  void start_service() {}
  void await_mailbox(int word, uint32_t expected);
  int submit(const int* descriptors, bool final_command);
};
bool Completion::load(std::memory_order) {
  if (++loads == 1) {
    // This callback runs after the first ACK load observed the old value.
    __atomic_store_n(core->box() + 33, uint32_t(result), __ATOMIC_RELEASE);
    if (publish_ack)
      __atomic_store_n(core->box() + 32, core->generation, __ATOMIC_RELEASE);
    core->service_status.store(status, std::memory_order_release);
  }
  return true;
}

@PRODUCTION_METHODS@

int main(int argc, char** argv) try {
  assert(argc == 2);
  const std::string test = argv[1];
  NativeCore core;
  int descriptors[14]{};
  if (test == "already_complete") {
    core.generation = 7;
    core.words[32] = 7;
    core.await_mailbox(32, 7);
    assert(core.service_done.loads == 0);
  } else if (test == "missing_ack_zero" || test == "missing_ack_error") {
    core.generation = 7;
    core.service_done.publish_ack = false;
    core.service_done.status = test == "missing_ack_error" ? 17 : 0;
    bool failed = false;
    try {
      core.await_mailbox(32, 7);
    } catch (const std::runtime_error& error) {
      failed = true;
      assert(std::string(error.what()) == "native service exited: " +
          std::to_string(core.service_done.status));
    }
    assert(failed);
  } else {
    const bool final = test != "nonfinal_kernel_error";
    const int command_result = test == "kernel_error" ||
        test == "nonfinal_kernel_error" ? 22 : 0;
    const int service_result = test == "rpc_error" ? 17 : 0;
    core.service_done.result = command_result;
    core.service_done.status = service_result;
    const int result = core.submit(descriptors, final);
    assert(result == (command_result ? command_result : service_result));
    assert(core.mailbox_commands == 1 && core.generation == 1);
    assert(core.service_done.loads >= 1);
  }
  std::printf("PASS %s\n", argv[1]);
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
"""


def method(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def native_mailbox(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++17 compiler is required")
    source = (ROOT / "runner/decode/native_core.cpp").read_text()
    await_source = method(source, "void NativeCore::await_mailbox(")
    submit_source = method(source, "int NativeCore::submit(")
    await_source = await_source.replace(
        'asm volatile("yield");', "std::this_thread::yield();"
    )
    recheck = "      if (__atomic_load_n(box() + word, __ATOMIC_ACQUIRE) != expected)\n"
    assert await_source.count(recheck) == 1
    directory = tmp_path_factory.mktemp("native-mailbox")
    binaries = {}
    for name, wait in [
        ("fixed", await_source),
        ("negative_control", await_source.replace(recheck, "", 1)),
    ]:
        cpp = directory / f"{name}.cpp"
        binary = directory / name
        cpp.write_text(
            STUB.replace("@PRODUCTION_METHODS@", wait + "\n" + submit_source)
        )
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-UNDEBUG",
                "-pthread",
                str(cpp),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        binaries[name] = binary
    return binaries


@pytest.mark.parametrize(
    "case",
    [
        "final_ok",
        "kernel_error",
        "rpc_error",
        "nonfinal_kernel_error",
        "missing_ack_zero",
        "missing_ack_error",
        "already_complete",
    ],
)
def test_native_mailbox_completion_and_status(native_mailbox, case):
    result = subprocess.run(
        [str(native_mailbox["fixed"]), case],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert f"PASS {case}" in result.stdout


def test_native_mailbox_negative_control_reproduces_false_exit(native_mailbox):
    result = subprocess.run(
        [str(native_mailbox["negative_control"]), "final_ok"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "native service exited: 0" in result.stderr
