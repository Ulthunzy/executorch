"""Explicit model registration; shared workflows never import model internals."""

from importlib import import_module
from executorch.examples.qualcomm.oss_scripts.moe.model.base import ModelAdapter


_models: dict[str, type[ModelAdapter]] = {}


def register_model(name, adapter):
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(adapter, type)
        or not issubclass(adapter, ModelAdapter)
    ):
        raise ValueError("A model name and a ModelAdapter subclass are required")
    if name in _models:
        raise ValueError(f"Model already registered: {name}")
    _models[name] = adapter


def create_model(name="qwen35"):
    if not isinstance(name, str) or not name:
        raise ValueError("A non-empty model_family string is required")
    if name == "qwen35" and name not in _models:
        import_module("executorch.examples.qualcomm.oss_scripts.moe.model.qwen35")
    if name not in _models:
        raise ValueError(f"Unsupported MoE model: {name}")
    return _models[name]()
