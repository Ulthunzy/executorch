from contextlib import contextmanager
from dataclasses import dataclass
import re
import subprocess
import codecs
import os


@dataclass(frozen=True)
class DevicePaths:
    root: str
    shared_model: str

    @property
    def prefill(self):
        return self.root + "/prefill"

    @property
    def decode(self):
        return self.root + "/decode"


class AndroidDevice:
    def __init__(self, serial):
        self.serial = serial

    def adb(self, *items):
        return subprocess.run(
            ["adb", "-s", self.serial, *map(str, items)],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        ).stdout

    def snapshot(self):
        thermal = self.adb("shell", "dumpsys thermalservice")
        match = re.search(r"Thermal Status:\s*(\d+)", thermal)
        return dict(
            thermal_status=int(match.group(1)) if match else None,
            meminfo=self.adb("shell", "cat /proc/meminfo"),
        )

    def check_target(self, target):
        actual = self.adb("shell", "getprop ro.soc.model").strip().upper()
        if not actual.startswith(target.soc_model):
            raise ValueError(
                f"Selected {target.soc_model}/{target.dsp_arch}; device reports ro.soc.model={actual!r}"
            )
        return actual

    def terminal(self, command, log_path):
        process = subprocess.Popen(
            ["adb", "-s", self.serial, "shell", "-T", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            with log_path.open("wb") as log:
                while True:
                    data = os.read(process.stdout.fileno(), 4096)
                    if not data:
                        break
                    log.write(data)
                    log.flush()
                    print(decoder.decode(data), end="", flush=True)
                print(decoder.decode(b"", final=True), end="", flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()

    @contextmanager
    def stream(self, command):
        process = subprocess.Popen(
            ["adb", "-s", self.serial, "shell", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            yield process
        finally:
            # Token validation can abort iteration while adb is still running.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
