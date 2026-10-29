import importlib
import importlib.util
from pathlib import Path
import sys


def load_binding(path: Path | None):
    name = "executorch.backends.qualcomm.python.PyQnnManagerAdaptor"
    if path is None:
        return importlib.import_module(name)
    path = path.resolve(strict=True)
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError(
                "Load the selected QNN binding before importing Qualcomm modules"
            )
        return module
    parent = importlib.import_module("executorch.backends.qualcomm.python")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent.PyQnnManagerAdaptor = module
    return module
