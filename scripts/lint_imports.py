#!/usr/bin/env python3
"""
scripts/lint_imports.py
Domain Boundary & Memory Safety Enforcement — 物理的制約

Hexagonal Architecture for ML:
1. NumPy/JAX import境界を自動監視。
2. `np.asarray`, `jnp.asarray`, `np.array`, `jnp.array` は登録済み array boundary 外で禁止。
3. `.astype()` と `.copy()` はメモリ倍増 (OOM) を防ぐため原則禁止。
4. 生の配列型 (`jnp.ndarray`, `Float64`等) の使用を禁止し、`JaxF64` / `NpF64` を強制。

ruff and pyrefly で捕捉できないドメイン違反やメモリ安全性の問題を検出するためのカスタムスクリプト。

Usage:
    uv run python scripts/lint_imports.py
"""
import re
import sys
from pathlib import Path

# ===== Domain Registry =====
# Mapper files: BOTH np and jax imports explicitly allowed (boundary converters) and pipelines
MAPPERS = {
    "core/types.py",
    "utils/batched_compute.py",
    "pipelines/strategies.py",
    "pipelines/components/executor.py",
    "pipelines/components/reporter.py",
    "pipelines/evaluation.py"
}

CONDITIONAL_JAX_OK = {}

ARRAY_CONVERSION_BOUNDARIES = {
    "core/types.py",
    "utils/batched_compute.py",
    "data/generators.py",
    "utils/gpu_utils.py",
    "readout/ridge.py",
    "models/reservoir/quantum/backend.py",
    "models/reservoir/quantum/functional.py",
    "models/reservoir/quantum/model.py",
    "pipelines/strategies.py",
}

CAST_BOUNDARIES = {
    "data/loaders.py",
    "models/generative.py",
    "models/nn/base.py",
    "models/reservoir/quantum/functional.py",
    "pipelines/components/executor.py",
    "pipelines/components/reporter.py",
    "pipelines/strategies.py",
}

SRC_ROOT = Path(__file__).parent.parent / "src" / "reservoir"

# 1. Imports (Boundary Check)
_import_np = re.compile(r"^\s*(import numpy|from numpy|import numpy\.)")
_import_jax = re.compile(r"^\s*(import jax\b|from jax\b|import jaxlib)")
_import_types = re.compile(r"from reservoir\.core\.types import (.*)")
_import_typing_cast = re.compile(r"^\s*from\s+typing\s+import\b.*\bcast\b")

# 2. Project Specific Types (Enforce NpF64 / JaxF64)
# Note: Pyrefly allows 'object', so we manually block it as an escape hatch.
_forbidden_object = re.compile(r"\bobject\b")
_forbidden_raw_float = re.compile(r"\b(?:Float64|Float32|Float)\[|\b(?:Float64|Float32|Float)\b(?!\w|\[)")
_forbidden_raw_ndarray = re.compile(r"\b(?:jnp\.ndarray|np\.ndarray|jax\.Array|jax\.numpy\.ndarray)\b")

# 3. Memory Safety & Array Creation (OOM Prevention)
_forbidden_asarray = re.compile(r"\b(?:np|jnp)\.asarray\b")
_forbidden_array_constructor = re.compile(r"\b(?:np|jnp)\.array\s*\(")
_forbidden_astype = re.compile(r"\.astype\(")
_forbidden_copy = re.compile(r"\.copy\(")

# 4. Callable Strictness (Pyrefly allows ellipsis `...`, but we forbid it for strictness)
_forbidden_callable_ellipsis = re.compile(r"\bCallable\[\s*\.\.\.")

# 5. Type Checking Escape Hatches
_forbidden_kwargs = re.compile(r"\bdef\s+\w+\s*\(.*(?:\*args|\*\*kwargs)")
_forbidden_cast_any = re.compile(r"cast\s*\(\s*Any\s*,")
_forbidden_cast_call = re.compile(r"(?:\btyping\.cast\s*\(|\bcast\s*\()")
_bare_type_ignore = re.compile(r"#\s*type:\s*ignore(?!\[)")

_forbidden_defensive_isinstance = re.compile(r"isinstance\s*\([^,]+,\s*\(?\s*(?:dict|list|tuple|int|float|str|bool)\b")


def check_file(path: Path) -> list[str]:
    """Return list of violation messages for a file."""
    rel = str(path.relative_to(SRC_ROOT))
    if path.name == "__init__.py" or "__pycache__" in str(path):
        return []

    lines = path.read_text().splitlines()
    violations = []
    has_np = False
    has_jax = False
    imported_jaxf64 = False
    imported_npf64 = False

    for i, line in enumerate(lines, 1):
        if _bare_type_ignore.search(line):
            violations.append(
                f"L{i}: ❌ Rule 13: Bare '# type: ignore' is forbidden. Use a specific code and register it."
            )

        if line.strip().startswith("#"):
            continue

        # Rule 1: Import Boundaries
        if _import_np.match(line):
            has_np = True
        if _import_jax.match(line):
            has_jax = True

        # Rule 2: Don't import both NpF64 and JaxF64 outside of Mappers
        types_match = _import_types.search(line)
        if types_match:
            imported = types_match.group(1)
            has_jaxf64 = re.search(r"\bJaxF64\b", imported) is not None
            has_npf64 = re.search(r"\bNpF64\b", imported) is not None
            imported_jaxf64 = imported_jaxf64 or has_jaxf64
            imported_npf64 = imported_npf64 or has_npf64

            if has_jaxf64 and has_npf64 and rel not in MAPPERS:
                violations.append(f"L{i}: ❌ Rule 7: Cannot import both JaxF64 and NpF64 outside Mapper.")

        # Rule 3: Memory Safety
        if _forbidden_asarray.search(line) and rel not in ARRAY_CONVERSION_BOUNDARIES:
            violations.append(
                f"L{i}: ❌ Rule 3: 'np.asarray', 'jnp.asarray' forbidden outside array boundary modules.")

        if _forbidden_array_constructor.search(line) and rel not in ARRAY_CONVERSION_BOUNDARIES:
            violations.append(
                f"L{i}: ❌ Rule 3: 'np.array', 'jnp.array' forbidden outside array boundary modules.")

        if _forbidden_astype.search(line) and rel not in CONDITIONAL_JAX_OK:
            violations.append(
                f"L{i}: ❌ Rule 9: '.astype()' forbidden (Fail Fast). Data must be loaded as np.float64 inherently.")

        if _forbidden_copy.search(line):
            violations.append(f"L{i}: ❌ Rule 10: '.copy()' forbidden (OOM Risk). Operations must be in-place.")

        # Rule 4: Ban 'object' as an escape hatch
        if _forbidden_object.search(line):
            if not re.search(r"class\s+\w+\s*\(object\):", line) and "logger" not in line:
                if re.search(r":\s*object|->\s*object|\[object\]", line):
                    violations.append(f"L{i}: ❌ Rule 2: 'object' type hint is a prohibited escape hatch.")

        # Rule 5: Raw Array Definitions
        if rel != "core/types.py" and _forbidden_raw_float.search(line):
            violations.append(f"L{i}: ❌ Rule 1: Raw 'Float64/Float32' forbidden. Use 'JaxF64' or 'NpF64'.")

        if rel != "core/types.py" and _forbidden_raw_ndarray.search(line):
            if re.search(r":\s*|->\s*", line):
                violations.append(f"L{i}: ❌ Rule 1/3: Raw array usage in type hint. Use 'JaxF64' or 'NpF64'.")

        # Rule 6: Callable strictness
        if _forbidden_callable_ellipsis.search(line):
            violations.append(f"L{i}: ❌ Rule 8: 'Callable[...,]' with ellipsis is forbidden. Specify exact arguments.")

        # check_file 関数の forループ内に追加
        if _forbidden_defensive_isinstance.search(line):
            # Unionの解決など、どうしても必要な場合は明示的な抑制コメントで回避させる設計にするか、
            # そもそも設計を見直させる
            violations.append(
                f"L{i}: ❌ Rule 14: Defensive 'isinstance' against basic types is forbidden. "
                "Trust Pyrefly & Beartype. (Use Type Narrowing only for explicit Unions)"
            )

        # check_file の forループ内
        if _forbidden_kwargs.search(line):
            violations.append(
                f"L{i}: ❌ Rule 12: '*args' and '**kwargs' are forbidden. Define all arguments explicitly for strict typing.")

        if _forbidden_cast_any.search(line):
            violations.append(f"L{i}: ❌ Rule 13: 'cast(Any, ...)' is strictly forbidden. It is a dangerous escape hatch that destroys type safety.")

        if (_import_typing_cast.search(line) or _forbidden_cast_call.search(line)) and rel not in CAST_BOUNDARIES:
            violations.append(
                f"L{i}: ❌ Rule 13: 'typing.cast' is forbidden outside registered unsafe boundaries."
            )

    # Boundary Violation Check (Numpy + JAX in same file)
    if has_np and has_jax and rel not in MAPPERS and rel not in CONDITIONAL_JAX_OK:
        violations.append("❌ BOUNDARY VIOLATION: Imports BOTH numpy AND jax (not a registered Mapper)")

    if imported_npf64 and imported_jaxf64 and rel not in MAPPERS:
        violations.append("❌ TYPE BOUNDARY VIOLATION: Imports BOTH NpF64 AND JaxF64 outside Mapper")

    if violations:
        return [f"\n📄 {rel}:"] + violations

    return []


def main():
    violations = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        violations.extend(check_file(py_file))

    if violations:
        print("=" * 80)
        print("🚨 DOMAIN & MEMORY SAFETY VIOLATIONS DETECTED 🚨")
        print("=" * 80)
        for v in violations:
            print(v)
        print(f"\nTotal Files with Violations: {len([v for v in violations if v.lstrip().startswith('📄')])}")
        print("\nACTION REQUIRED: Fix boundary or memory violations as defined in AGENT.md")
        sys.exit(1)
    else:
        print("✅ All domains and memory operations are clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
