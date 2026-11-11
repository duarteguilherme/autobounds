"""Load optional feature dependencies without masking broken installations."""
from importlib import import_module


def require_optional(module, extra):
    """Import a feature dependency, explaining how to install it if absent."""
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + '.')):
            raise ImportError(
                f"This feature requires the '{extra}' extra. "
                f"Install it with: python -m pip install 'autobounds[{extra}]'"
            ) from exc
        raise
