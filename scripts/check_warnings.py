"""Surface Python warnings while importing Reservoir project modules."""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_NAME = "reservoir"


def _prepare_import_path() -> None:
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _iter_project_modules() -> list[str]:
    package = importlib.import_module(PACKAGE_NAME)
    module_names = [package.__name__]
    if not hasattr(package, "__path__"):
        return module_names

    module_path = package.__path__
    for module_info in pkgutil.walk_packages(module_path, prefix=f"{package.__name__}."):
        module_names.append(module_info.name)
    return sorted(module_names)


def _format_warning(module_name: str, warning: warnings.WarningMessage) -> str:
    return (
        f"{module_name}: {warning.filename}:{warning.lineno}: "
        f"{warning.category.__name__}: {warning.message}"
    )


def main() -> int:
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _prepare_import_path()

    warning_lines: list[str] = []
    import_errors: list[str] = []

    for module_name in _iter_project_modules():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            try:
                importlib.import_module(module_name)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
                continue

        if caught is None:
            continue

        for warning in caught:
            warning_lines.append(_format_warning(module_name, warning))

    if warning_lines:
        print("Python warnings:")
        for line in warning_lines:
            print(f"- {line}")
    else:
        print("No Python warnings emitted while importing Reservoir project modules.")

    if import_errors:
        print("\nImport errors:")
        for line in import_errors:
            print(f"- {line}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
