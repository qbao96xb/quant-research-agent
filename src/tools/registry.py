# src/tools/registry.py
import importlib
import pkgutil
from pathlib import Path

from pydantic import ValidationError

from .errors import ToolError
from .registry_core import REGISTRY, tool  # noqa: F401
from . import cache  # noqa: F401  (registers list_artifacts if decorated)

# Import every *_tools.py module so its @tool decorators run
for _m in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if _m.name.endswith("_tools"):
        importlib.import_module(f"{__package__}.{_m.name}")


def call(name: str, **kwargs) -> dict:
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolError("UNKNOWN_TOOL", f"No tool named {name!r}",
                         f"Options: {sorted(REGISTRY)}").to_dict()
    try:
        return spec["fn"](spec["args"](**kwargs))
    except ToolError as e:
        return e.to_dict()
    except ValidationError as e:
        return ToolError("BAD_ARGUMENT", str(e.errors()[0]["msg"]),
                         f"Field: {e.errors()[0]['loc']}").to_dict()
    except Exception as e:
        return ToolError("INTERNAL", f"{type(e).__name__}: {e}").to_dict()


def schemas() -> list[dict]:
    return [dict(name=n, description=s["fn"].__doc__, parameters=s["args"].model_json_schema())
            for n, s in REGISTRY.items()]
