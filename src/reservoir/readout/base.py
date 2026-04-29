"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/readout/base.py
ABC for all readout components (Step 7).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, ConfigDict



class ReadoutModule(ABC):
    """Abstract base for readout components (e.g., ridge regression, FNN)."""

    hidden_layers: tuple[int, ...] | None = None

    @abstractmethod
    def fit(self, states: JaxF64, targets: JaxF64) -> ReadoutModule:
        """Fit the readout on reservoir states and target labels."""

    @abstractmethod
    def predict(self, states: JaxF64) -> JaxF64:
        """Predict from reservoir states."""

    @abstractmethod
    def to_dict(self) -> ConfigDict:
        """Serialize to config dict."""


__all__ = ["ReadoutModule"]
