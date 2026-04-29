"""Pytest bootstrap for deterministic JAX unit-test imports."""

from __future__ import annotations

import os


os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
