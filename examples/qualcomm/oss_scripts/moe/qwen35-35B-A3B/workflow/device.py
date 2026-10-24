from contextlib import contextmanager
from dataclasses import dataclass
import re
import subprocess


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
