"""Dependency-driven export with locked, content-checked checkpoints."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from executorch.examples.qualcomm.oss_scripts.moe.checkpoint import (
    StageCheckpoint,
    artifact_stamps,
    remove_path,
    write_json,
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    cleanup: tuple[Path, ...] = ()
    always_run: bool = False
    preserve_outputs: bool = False
    requires: tuple[str, ...] | None = None
    publishes: tuple[str, ...] = ()
    resource: str = "cpu"
    needs_device: bool = False


class ExportWorkflow(ABC):
    def __init__(self, config, root):
        self.config = config
        self.root = Path(root)
        self.settings = config.model_settings
        self.stages = self.plan()
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names) or "package" in names:
            raise ValueError(
                "Export stage names must be unique and cannot be 'package'"
            )
        self.dependencies = {}
        by_name = {stage.name: stage for stage in self.stages}
        for index, stage in enumerate(self.stages):
            dependencies = stage.requires
            if dependencies is None:
                dependencies = (names[index - 1],) if index else ()
            for dependency in dependencies:
                producer, _, milestone = dependency.partition(":")
                if producer not in by_name or (
                    milestone and milestone not in by_name[producer].publishes
                ):
                    raise ValueError(
                        f"Unknown dependency {dependency} for {stage.name}"
                    )
            self.dependencies[stage.name] = dependencies
        pending = set(names)
        while pending:
            ready = {
                name
                for name in pending
                if not any(
                    dep.partition(":")[0] in pending for dep in self.dependencies[name]
                )
            }
            if not ready:
                raise ValueError("Export stage dependencies contain a cycle")
            pending -= ready

    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def check(self):
        pass

    @abstractmethod
    def environment(self):
        pass

    @abstractmethod
    def identity(self):
        pass

    @abstractmethod
    def protected_paths(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    def package_path(self):
        return None

    def preserved_paths(self):
        return ()

    def package_identity(self):
        return {}

    def package(self):
        raise NotImplementedError("This model does not implement packaging")

    def describe(self):
        return json.dumps(
            dict(
                configuration=self.config.to_dict(),
                stages=[
                    dict(
                        name=s.name,
                        command=list(s.command),
                        outputs=list(map(str, s.outputs)),
                        cleanup=list(map(str, s.cleanup)),
                        requires=list(self.dependencies[s.name]),
                        publishes=list(s.publishes),
                        resource=s.resource,
                        needs_device=s.needs_device,
                    )
                    for s in self.stages
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )

    def run(self, resume=True):
        self.check()
        out = self.config.output
        with StageCheckpoint(
            out,
            self.identity(),
            "EXPORT_STATE.json",
            self.protected_paths(),
            preserved=self.preserved_paths(),
        ) as cache:
            write_json(out / "MODEL_CONFIG.json", self.settings.to_dict())
            write_json(out / "EXPORT_PLAN.json", json.loads(self.describe()))
            cache.dirty = not resume
            self.execute_stages(cache)
            report = self.finish()
            package = self.package_path()
            if package is not None:
                signature = self.package_identity()
                previous = cache.state["stages"].get("package", {})
                if any(
                    previous.get(k) != v for k, v in signature.items()
                ) or not cache.reusable("package", (package,)):
                    print("EXPORT_START package", flush=True)
                    cache.update("package", status="running")
                    try:
                        self.package()
                        cache.update(
                            "package",
                            status="complete",
                            outputs=artifact_stamps((package,)),
                            **signature,
                        )
                    except BaseException as error:
                        cache.update("package", status="failed", error=str(error))
                        raise
                    print("EXPORT_DONE package", flush=True)
                else:
                    print("EXPORT_REUSED package", flush=True)
                report["model_pte"] = str(package)
                report["delivery"] = (
                    "single PTE; model resources are embedded as named data"
                )
            write_json(out / "EXPORT.json", report)
        print("EXPORT_COMPLETE", out, flush=True)
        return out / "EXPORT.json"

    def execute_stages(self, cache):
        out = self.config.output
        completed, available = set(), set()
        pending = []
        # Compute reuse in plan order before concurrent work changes the checkpoint.
        for stage in self.stages:
            if cache.reusable(stage.name, stage.outputs) and not stage.always_run:
                completed.add(stage.name)
                available.update(f"{stage.name}:{name}" for name in stage.publishes)
                print("EXPORT_REUSED", stage.name, flush=True)
            else:
                pending.append(stage)
        running = {}
        selector = selectors.DefaultSelector()
        try:
            while pending or running:
                for stage in tuple(pending):
                    if len(running) == 2:
                        break
                    if any(
                        item["stage"].resource == stage.resource
                        for item in running.values()
                    ) or not all(
                        dep in completed or dep in available
                        for dep in self.dependencies[stage.name]
                    ):
                        continue
                    (out / "EXPORT.json").unlink(missing_ok=True)
                    if not stage.preserve_outputs:
                        for path in (*stage.outputs, *stage.cleanup):
                            if path.is_relative_to(out):
                                remove_path(path)
                    log = out / "logs" / f"{stage.name}.log"
                    log.parent.mkdir(exist_ok=True)
                    stream = log.open("w")
                    reader, writer = os.pipe()
                    os.set_blocking(reader, False)
                    env = self.environment()
                    env["ET_EXPORT_PROGRESS_FD"] = str(writer)
                    cache.update(stage.name, status="running", log=str(log))
                    print("EXPORT_START", stage.name, "log=" + str(log), flush=True)
                    try:
                        process = subprocess.Popen(
                            stage.command,
                            cwd=self.root,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            pass_fds=(cache.lock.fileno(), writer),
                            start_new_session=True,
                        )
                    except BaseException as error:
                        cache.update(
                            stage.name, status="failed", error=str(error), log=str(log)
                        )
                        stream.close()
                        os.close(reader)
                        raise
                    finally:
                        os.close(writer)
                    running[stage.name] = dict(
                        stage=stage,
                        process=process,
                        log=log,
                        stream=stream,
                        reader=reader,
                        buffer=b"",
                        started=time.monotonic(),
                    )
                    selector.register(reader, selectors.EVENT_READ, stage.name)
                    pending.remove(stage)
                if not running:
                    raise RuntimeError("Export has unresolved artifact dependencies")
                for key, _ in selector.select(timeout=0.1):
                    item = running[key.data]
                    self.read_progress(selector, item, available)
                for name, item in tuple(running.items()):
                    result = item["process"].poll()
                    if result is None:
                        continue
                    # Drain remaining progress before closing an exited producer's pipe.
                    if item["reader"] in selector.get_map():
                        self.read_progress(selector, item, available)
                    if item["buffer"]:
                        raise ValueError(f"Incomplete progress event from {name}")
                    if result:
                        raise RuntimeError(
                            f"{name} exited with {result}; see {item['log']}"
                        )
                    stage = item["stage"]
                    for path in stage.cleanup:
                        path.unlink(missing_ok=True)
                    cache.update(
                        name,
                        status="complete",
                        outputs=artifact_stamps(stage.outputs),
                        seconds=time.monotonic() - item["started"],
                        log=str(item["log"]),
                    )
                    completed.add(name)
                    available.update(f"{name}:{value}" for value in stage.publishes)
                    self.close_stage(selector, item)
                    del running[name]
                    print("EXPORT_DONE", name, flush=True)
        except BaseException as error:
            for name, item in running.items():
                cache.update(
                    name, status="failed", error=str(error), log=str(item["log"])
                )
            raise
        finally:
            for item in running.values():
                try:
                    os.killpg(item["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 3
            for item in running.values():
                try:
                    item["process"].wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(item["process"].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    item["process"].wait()
                self.close_stage(selector, item)
            selector.close()

    @staticmethod
    def read_progress(selector, item, available):
        reader = item["reader"]
        name = item["stage"].name
        while True:
            try:
                data = os.read(reader, 65536)
            except BlockingIOError:
                return
            item["buffer"] += data
            while b"\n" in item["buffer"]:
                line, item["buffer"] = item["buffer"].split(b"\n", 1)
                message = json.loads(line)
                milestone = message.get("ready")
                if milestone is not None:
                    if milestone not in item["stage"].publishes:
                        raise ValueError(f"Unexpected export milestone: {milestone}")
                    available.add(f"{name}:{milestone}")
                    print("EXPORT_READY", name, milestone, flush=True)
                if "metric" in message:
                    print(message["metric"], flush=True)
            if not data:
                selector.unregister(reader)
                return

    @staticmethod
    def close_stage(selector, item):
        if item["reader"] in selector.get_map():
            selector.unregister(item["reader"])
        os.close(item["reader"])
        item["stream"].close()
