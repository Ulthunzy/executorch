"""Inference lifecycle shared by model adapters and device frontends."""

from abc import ABC, abstractmethod
import json


class InferenceBackend(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def validate_artifacts(self):
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def check_plans(self):
        pass

    @abstractmethod
    def run_chat(self):
        pass

    @abstractmethod
    def run_diagnostic(self, metadata):
        pass


class MoeRuntime:
    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def run(self, *, validate=True):
        args = self.backend.args
        if args.print_config:
            print(
                json.dumps(args.model_settings.to_dict(), ensure_ascii=False, indent=2)
            )
            return
        if validate:
            self.backend.validate_artifacts()
        metadata = self.backend.prepare()
        if args.check:
            self.backend.check_plans()
            print(f"Local artifacts and optimized runners ready: {args.artifacts}")
            return
        if args.runner == "diagnostic":
            return self.backend.run_diagnostic(metadata)
        if args.runner != "chat":
            raise ValueError(f"Unsupported inference frontend: {args.runner}")
        return self.backend.run_chat()
