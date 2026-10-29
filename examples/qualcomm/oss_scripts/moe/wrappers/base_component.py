"""Model-independent serial export with locked, content-checked checkpoints."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
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
            for stage in self.stages:
                if cache.reusable(stage.name, stage.outputs) and not stage.always_run:
                    print("EXPORT_REUSED", stage.name, flush=True)
                    continue
                (out / "EXPORT.json").unlink(missing_ok=True)
                if not stage.preserve_outputs:
                    for path in (*stage.outputs, *stage.cleanup):
                        if path.is_relative_to(out):
                            remove_path(path)
                log = out / "logs" / f"{stage.name}.log"
                log.parent.mkdir(exist_ok=True)
                started = time.monotonic()
                cache.update(stage.name, status="running", log=str(log))
                print("EXPORT_START", stage.name, "log=" + str(log), flush=True)
                try:
                    with log.open("w") as stream:
                        result = subprocess.run(
                            stage.command,
                            cwd=self.root,
                            env=self.environment(),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            pass_fds=(cache.lock.fileno(),),
                        )
                    if result.returncode:
                        raise RuntimeError(
                            f"{stage.name} exited with {result.returncode}; see {log}"
                        )
                    for path in stage.cleanup:
                        path.unlink(missing_ok=True)
                    cache.update(
                        stage.name,
                        status="complete",
                        seconds=time.monotonic() - started,
                        outputs=artifact_stamps(stage.outputs),
                        log=str(log),
                    )
                except BaseException as error:
                    cache.update(
                        stage.name, status="failed", error=str(error), log=str(log)
                    )
                    raise
                print("EXPORT_DONE", stage.name, flush=True)
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
