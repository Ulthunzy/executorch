from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def root(self) -> Path:
        pass

    @property
    @abstractmethod
    def default_export_config(self) -> Path:
        pass

    @abstractmethod
    def load_export_config(self, path, output=None, **overrides):
        pass

    @abstractmethod
    def create_export_workflow(self, config):
        pass

    @abstractmethod
    def parse_runtime_args(self, argv):
        pass

    @abstractmethod
    def create_inference_backend(self, args):
        pass


@dataclass(frozen=True)
class ModelArchitecture:
    layers: int
    group_size: int
    hidden_size: int
    experts: int
    top_k: int

    @property
    def groups(self):
        return (self.layers + self.group_size - 1) // self.group_size
