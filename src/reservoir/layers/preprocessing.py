"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/preprocessing.py
STEP2 Preprocessing layers.
Here we define preprocessing which doesn't change any dimensions, such as scaling.
"""
from __future__ import annotations
import abc
from functools import singledispatch

from beartype import beartype
import numpy as np
from typing import TYPE_CHECKING

from reservoir.core.types import NpF64  # noqa: TC001

if TYPE_CHECKING:
    from reservoir.core.types import ConfigDict
    from reservoir.models.config import (
        AffineScalerConfig,
        BoundedAffineScalerConfig,
        MinMaxScalerConfig,
        PreprocessingConfig,
        RawConfig,
        StandardScalerConfig,
    )


# --- 1. Interface Definition ---

@beartype
class Preprocessor(abc.ABC):
    """
    Abstract Base Class for Preprocessing Layers.
    Defines the contract that all preprocessing layers must follow.
    Preprocessors do NOT change dimensions.
    """

    @abc.abstractmethod
    def fit(self, X: NpF64) -> Preprocessor:
        """Fit the preprocessor on training data."""

    @abc.abstractmethod
    def transform(self, X: NpF64) -> NpF64:
        """Apply the transformation."""

    @abc.abstractmethod
    def inverse_transform(self, X: NpF64) -> NpF64:
        """Reverse the transformation."""

    @abc.abstractmethod
    def to_dict(self) -> ConfigDict:
        """Serialize configuration parameters."""

    def fit_transform(self, X: NpF64) -> NpF64:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def __call__(self, X: NpF64) -> NpF64:
        """Alias for transform."""
        return self.transform(X)

    @property
    def label(self) -> str:
        """Short label for filenames."""
        return self.__class__.__name__


# --- 2. Concrete Implementations ---

@beartype
class StandardScaler(Preprocessor):
    """Standard scaler (mean removal and variance scaling)."""

    def __init__(self):
        self.mean_: NpF64 | None = None
        self.scale_: NpF64 | None = None

    def fit(self, X: NpF64) -> StandardScaler:
        X_np = X

        if X_np.ndim == 3:
            reduce_axis = (0, 1)
        else:
            reduce_axis = 0

        self.mean_ = np.mean(X_np, axis=reduce_axis)
        self.scale_ = np.std(X_np, axis=reduce_axis)
        
        # Avoid division by zero and extreme scaling for near-constant features
        if self.scale_ is not None:
            self.scale_[self.scale_ < 1e-6] = 1.0
            
        return self

    def transform(self, X: NpF64) -> NpF64:
        arr = X
        mean = self.mean_
        scale = self.scale_
        if mean is None or scale is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        arr -= mean
        arr /= scale
        return arr

    def inverse_transform(self, X: NpF64) -> NpF64:
        arr = X
        mean = self.mean_
        scale = self.scale_
        if mean is None or scale is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform().")
        arr *= scale
        arr += mean
        return arr

    def to_dict(self) -> ConfigDict:
        return {
            "type": "standard_scaler",
        }

    @property
    def label(self) -> str:
        return "StandardScaler"


@beartype
class MinMaxScaler(Preprocessor):
    """
    Min-Max Scaler with feature range scaling.
    Scales data to [feature_min, feature_max].
    Formula: X_std = (X - X.min) / (X.max - X.min)
             X_scaled = X_std * (feature_max - feature_min) + feature_min
    """

    def __init__(self, feature_min: float , feature_max: float):
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.min_: NpF64 | None = None
        self.range_: NpF64 | None = None  # X_max - X_min

    def fit(self, X: NpF64) -> MinMaxScaler:
        X_np = X

        if X_np.ndim == 3:
            reduce_axis = (0, 1)
        else:
            reduce_axis = 0

        min_val = np.min(X_np, axis=reduce_axis)
        max_val = np.max(X_np, axis=reduce_axis)
        range_val = max_val - min_val
        self.min_ = min_val
        self.range_ = range_val

        # Avoid division by zero for constant features
        if self.range_ is not None:
            self.range_[self.range_ < 1e-12] = 1.0

        return self

    def transform(self, X: NpF64) -> NpF64:
        arr = X
        min_ = self.min_
        range_ = self.range_
        if min_ is None or range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform().")
        # 1. Scale to [0, 1]
        arr -= min_
        arr /= range_
        
        # 2. Scale to [feature_min, feature_max]
        scale = self.feature_max - self.feature_min
        arr *= scale
        arr += self.feature_min
        
        return arr

    def inverse_transform(self, X: NpF64) -> NpF64:
        arr = X
        min_ = self.min_
        range_ = self.range_
        if min_ is None or range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before inverse_transform().")
        # 1. Reverse Scale to [0, 1]
        scale = self.feature_max - self.feature_min
        if scale != 0:
            arr -= self.feature_min
            arr /= scale
            
        # 2. Reverse Scale to Original
        arr *= range_
        arr += min_
            
        return arr

    def to_dict(self) -> ConfigDict:
        return {
            "type": "min_max_scaler",
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
        }

    @property
    def label(self) -> str:
        return f"Min{self.feature_min:.2f}Max{self.feature_max:.2f}"


@beartype
class IdentityPreprocessor(Preprocessor):
    """No-op preprocessor for RAW mode."""

    def fit(self, X: NpF64) -> IdentityPreprocessor:
        _ = X  # Identity: no fitting needed
        return self

    def transform(self, X: NpF64) -> NpF64:
        return X

    def inverse_transform(self, X: NpF64) -> NpF64:
        return X

    def to_dict(self) -> ConfigDict:
        return {"type": "identity"}

    @property
    def label(self) -> str:
        return "identity"


@beartype
class AffineScaler(Preprocessor):
    """
    Affine transformation scaler.
    Formula: X_scaled = X * input_scale + shift
    """

    def __init__(self, input_scale: float, shift: float):
        self.input_scale = input_scale
        self.shift = shift

    def fit(self, X: NpF64) -> AffineScaler:
        # AffineScaler is stateless (parameters are provided at init), so fit does nothing.
        return self

    def transform(self, X: NpF64) -> NpF64:
        arr = X
        arr *= self.input_scale
        arr += self.shift
        return arr

    def inverse_transform(self, X: NpF64) -> NpF64:
        # Avoid division by zero
        if self.input_scale == 0:
            return X
            
        arr = X
        arr -= self.shift
        arr /= self.input_scale
        return arr

    def to_dict(self) -> ConfigDict:
        return {
            "type": "affine_scaler",
            "input_scale": self.input_scale,
            "shift": self.shift,
        }

    @property
    def label(self) -> str:
        return f"Affine_a{self.input_scale:.2f}_b{self.shift:.2f}"


@beartype
class BoundedAffineScaler(Preprocessor):
    """MinMax[-1,1] followed by a bounded affine transformation.

    Guarantees output ∈ [-bound, bound] for any valid parameter combination.

    Parameters
    ----------
    scale : float
        Contraction factor in (0, 1].
    relative_shift : float
        Shift proportion in [-1, 1].  The actual shift is
        ``relative_shift * bound * (1 - scale)`` so the affine image can never
        exceed [-bound, bound].
    bound : float
        Maximum absolute boundary limit.

    Computation:

    x_norm = MinMax(X, feature_min=-1, feature_max=1)
    shift  = relative_shift * bound * (1 - scale)
    y      = (scale * bound) * x_norm + shift        # ∈ [-bound, bound] always

    min = (scale * bound) * (-1) + shift = -scale * bound + relative_shift * bound * (1 - scale)
    max = (scale * bound) * (1) + shift = scale * bound + relative_shift * bound * (1 - scale)
    """

    def __init__(self, scale: float, relative_shift: float, bound: float = 1.0):
        self.scale = float(scale)
        self.relative_shift = float(relative_shift)
        self.bound = float(bound)
        # Compute the actual shift from the relative parameter
        self._shift = self.relative_shift * self.bound * (1.0 - self.scale)

        # Internal stages: reuse existing implementations
        self._minmax = MinMaxScaler(feature_min=-1.0, feature_max=1.0)
        self._affine = AffineScaler(input_scale=self.scale * self.bound, shift=self._shift)

    def fit(self, X: NpF64) -> BoundedAffineScaler:
        self._minmax.fit(X)
        # AffineScaler.fit is a no-op (stateless), but call for API consistency
        self._affine.fit(X)
        return self

    def transform(self, X: NpF64) -> NpF64:
        arr = self._minmax.transform(X)
        return self._affine.transform(arr)

    def inverse_transform(self, X: NpF64) -> NpF64:
        arr = self._affine.inverse_transform(X)
        return self._minmax.inverse_transform(arr)

    def to_dict(self) -> ConfigDict:
        return {
            "type": "bounded_affine_scaler",
            "scale": self.scale,
            "relative_shift": self.relative_shift,
            "bound": self.bound,
        }

    @property
    def label(self) -> str:
        f_min = -(self.bound * self.scale) + self._shift
        f_max = (self.bound * self.scale) + self._shift
        return f"Min{f_min:.2f}Max{f_max:.2f}"


# --- 3. Factory Logic (Dependency Injection Helper) ---

@singledispatch
def create_preprocessor(config: PreprocessingConfig) -> Preprocessor:
    """
    Factory function to create a Preprocessor instance based on the config type.
    Raises TypeError if the config type is not registered.
    """
    raise TypeError(f"Unknown preprocessor config type: {config.__class__.__name__}")


def register_preprocessors(
    RawConfigClass: type[RawConfig],
    StandardScalerConfigClass: type[StandardScalerConfig],
    MinMaxScalerConfigClass: type[MinMaxScalerConfig] | None = None,
    AffineScalerConfigClass: type[AffineScalerConfig] | None = None,
    BoundedAffineScalerConfigClass: type[BoundedAffineScalerConfig] | None = None,
) -> None:
    """
    Register config classes with the factory.
    Call this once at module initialization.
    """

    @create_preprocessor.register(RawConfigClass)
    def _(_config: RawConfig) -> Preprocessor:
        return IdentityPreprocessor()

    @create_preprocessor.register(StandardScalerConfigClass)
    def _(_config: StandardScalerConfig) -> Preprocessor:
        return StandardScaler()

    if MinMaxScalerConfigClass is not None:
        @create_preprocessor.register(MinMaxScalerConfigClass)
        def _(config: MinMaxScalerConfig) -> Preprocessor:
            return MinMaxScaler(feature_min=config.feature_min, feature_max=config.feature_max)

    if AffineScalerConfigClass is not None:
        @create_preprocessor.register(AffineScalerConfigClass)
        def _(config: AffineScalerConfig) -> Preprocessor:
            return AffineScaler(input_scale=config.input_scale, shift=config.shift)

    if BoundedAffineScalerConfigClass is not None:
        @create_preprocessor.register(BoundedAffineScalerConfigClass)
        def _(config: BoundedAffineScalerConfig) -> Preprocessor:
            return BoundedAffineScaler(scale=config.scale, relative_shift=config.relative_shift, bound=config.bound)


__all__ = [
    "Preprocessor",
    "StandardScaler",
    "MinMaxScaler",
    "AffineScaler",
    "BoundedAffineScaler",
    "IdentityPreprocessor",
    "create_preprocessor",
    "register_preprocessors",
]
