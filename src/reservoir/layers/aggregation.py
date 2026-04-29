"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/aggregation.py
Step 6 aggregation layers.
State aggregation components — Strategy Pattern per AggregationMode.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod

from beartype import beartype
import jax.numpy as jnp
from reservoir.core.types import JaxF64, ConfigDict
from reservoir.core.types import to_np_f64
from reservoir.utils.reporting import print_feature_stats


# ==========================================
# Aggregation Mode Enum
# ==========================================

class AggregationMode(enum.StrEnum):
    """State aggregation strategies for sequence-to-feature reduction."""

    LAST = "last"
    MEAN = "mean"
    LAST_MEAN = "last_mean"
    MTS = "mts"
    CONCAT = "concat"
    SEQUENCE = "sequence"

    def __str__(self) -> str:
        return self.value


# ==========================================
# Abstract Base
# ==========================================

@beartype
class StateAggregator(ABC):
    """Abstract base for time-axis aggregation strategies."""

    def __init__(self, mode: AggregationMode) -> None:
        self.mode = mode

    # --- Interface required by Aggregator Protocol ---

    def fit(self, features: JaxF64, y: JaxF64 | None = None) -> StateAggregator:
        _ = (features, y)
        return self

    def transform(self, features: JaxF64, log_label: str | None = None) -> JaxF64:
        result = self._aggregate(features)

        if result.ndim != 2:
            raise ValueError(f"Aggregation output must be 2D, got {result.shape}")

        if log_label is not None:
            print_feature_stats(to_np_f64(result), "aggregation.py", log_label)
        return result

    def fit_transform(self, features: JaxF64) -> JaxF64:
        return self.transform(features)

    def __call__(self, features: JaxF64, log_label: str | None = None) -> JaxF64:
        return self.transform(features, log_label=log_label)

    # --- Abstract: each subclass implements its own reduction ---

    @abstractmethod
    def _aggregate(self, states: JaxF64) -> JaxF64:
        """Reduce (Batch, Time, Feat) or (Time, Feat) along the time axis."""

    @abstractmethod
    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        """Compute aggregated feature dimension without materializing data."""

    # --- Shared helpers ---

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute output shape from (Batch, Time, Units) or (Time, Units)."""
        if len(input_shape) == 3:
            batch, steps, units = input_shape
            return self._output_shape_3d(batch, steps, units)
        if len(input_shape) == 2:
            steps, units = input_shape
            return self._output_shape_2d(steps, units)
        raise ValueError(f"Unsupported input shape: {input_shape}")

    def _output_shape_3d(self, batch: int, steps: int, units: int) -> tuple[int, ...]:
        """Default 3D output shape: (Batch, output_dim)."""
        return batch, self.get_output_dim(units, steps)

    def _output_shape_2d(self, steps: int, units: int) -> tuple[int, ...]:
        """Default 2D output shape: (output_dim,)."""
        return (self.get_output_dim(units, steps),)

    def to_dict(self) -> ConfigDict:
        return {"mode": self.mode.value}

    @staticmethod
    def aggregate(states: JaxF64, mode: AggregationMode) -> JaxF64:
        """Static helper for functional contexts (backward-compatible)."""
        agg = create_aggregator(mode)
        return agg._aggregate(states)

    @classmethod
    def from_dict(cls, data: ConfigDict) -> StateAggregator:
        return create_aggregator(AggregationMode(str(data.get("mode", AggregationMode.LAST.value))))


# ==========================================
# Concrete Strategies
# ==========================================

@beartype
class LastAggregator(StateAggregator):
    """Take the last time step."""

    def __init__(self) -> None:
        super().__init__(AggregationMode.LAST)

    def _aggregate(self, states: JaxF64) -> JaxF64:
        if states.ndim == 3:
            return states[:, -1, :]
        if states.ndim == 2:
            return states[-1:]  # Keep 2D: (1, Feat)
        raise ValueError(f"Unsupported shape {states.shape}")

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        _validate_positive(n_units, n_steps)
        return n_units


@beartype
class MeanAggregator(StateAggregator):
    """Take the mean across time."""

    def __init__(self) -> None:
        super().__init__(AggregationMode.MEAN)

    def _aggregate(self, states: JaxF64) -> JaxF64:
        if states.ndim == 3:
            return jnp.mean(states, axis=1)
        if states.ndim == 2:
            return jnp.mean(states, axis=0, keepdims=True)  # (1, Feat)
        raise ValueError(f"Unsupported shape {states.shape}")

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        _validate_positive(n_units, n_steps)
        return n_units


@beartype
class LastMeanAggregator(StateAggregator):
    """Concatenate last state and mean state."""

    def __init__(self, mode: AggregationMode = AggregationMode.LAST_MEAN) -> None:
        super().__init__(mode)

    def _aggregate(self, states: JaxF64) -> JaxF64:
        if states.ndim == 3:
            last = states[:, -1, :]
            mean = jnp.mean(states, axis=1)
            return jnp.concatenate([last, mean], axis=1)
        if states.ndim == 2:
            last = states[-1:]
            mean = jnp.mean(states, axis=0, keepdims=True)
            return jnp.concatenate([last, mean], axis=1)
        raise ValueError(f"Unsupported shape {states.shape}")

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        _validate_positive(n_units, n_steps)
        return n_units * 2


@beartype
class MTSAggregator(StateAggregator):
    """Mixed Three States (MTS): join states at 1/3, 2/3, and the last step."""

    def __init__(self) -> None:
        super().__init__(AggregationMode.MTS)

    def _aggregate(self, states: JaxF64) -> JaxF64:
        time_axis = 1 if states.ndim == 3 else 0
        n_steps = states.shape[time_axis]
        
        idx1 = n_steps // 3
        idx2 = (2 * n_steps) // 3
        idx3 = -1
        
        if states.ndim == 3:
            s1 = states[:, idx1, :]
            s2 = states[:, idx2, :]
            s3 = states[:, idx3, :]
            return jnp.concatenate([s1, s2, s3], axis=1)
        if states.ndim == 2:
            s1 = states[idx1:idx1+1, :]
            s2 = states[idx2:idx2+1, :]
            s3 = states[idx3:, :]
            return jnp.concatenate([s1, s2, s3], axis=1)
        raise ValueError(f"Unsupported shape {states.shape}")

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        _validate_positive(n_units, n_steps)
        return n_units * 3


@beartype
class ConcatAggregator(StateAggregator):
    """Flatten all time steps into a single feature vector."""

    def __init__(self) -> None:
        super().__init__(AggregationMode.CONCAT)

    def _aggregate(self, states: JaxF64) -> JaxF64:
        if states.ndim == 3:
            return states.reshape(states.shape[0], -1)
        if states.ndim == 2:
            return states.reshape(1, -1)  # (1, Time*Feat)
        raise ValueError(f"Unsupported shape {states.shape}")

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        _validate_positive(n_units, n_steps)
        return n_units * n_steps

    def _output_shape_3d(self, batch: int, steps: int, units: int) -> tuple[int, ...]:
        return batch, steps * units

    def _output_shape_2d(self, steps: int, units: int) -> tuple[int, ...]:
        return (steps * units,)


@beartype
class SequenceAggregator(StateAggregator):
    """No reduction — flatten batch and time into samples."""

    def __init__(self) -> None:
        super().__init__(AggregationMode.SEQUENCE)

    def _aggregate(self, states: JaxF64) -> JaxF64:
        if states.ndim == 3:
            return states.reshape(-1, states.shape[-1])
        if states.ndim == 2:
            return states
        raise ValueError(f"Unsupported shape {states.shape}")

    def get_output_dim(self, n_units: int, n_steps: int) -> int:
        _validate_positive(n_units, n_steps)
        return n_units

    def _output_shape_3d(self, batch: int, steps: int, units: int) -> tuple[int, ...]:
        return batch * steps, units

    def _output_shape_2d(self, steps: int, units: int) -> tuple[int, ...]:
        return steps, units


# ==========================================
# Factory
# ==========================================

_REGISTRY: dict[AggregationMode, type[StateAggregator]] = {
    AggregationMode.LAST: LastAggregator,
    AggregationMode.MEAN: MeanAggregator,
    AggregationMode.LAST_MEAN: LastMeanAggregator,
    AggregationMode.MTS: MTSAggregator,
    AggregationMode.CONCAT: ConcatAggregator,
    AggregationMode.SEQUENCE: SequenceAggregator,
}


def create_aggregator(mode: AggregationMode) -> StateAggregator:
    """Factory: create the right aggregator for the given mode."""
    if mode is AggregationMode.LAST:
        return LastAggregator()
    if mode is AggregationMode.MEAN:
        return MeanAggregator()
    if mode is AggregationMode.LAST_MEAN:
        return LastMeanAggregator(mode=AggregationMode.LAST_MEAN)
    if mode is AggregationMode.MTS:
        return MTSAggregator()
    if mode is AggregationMode.CONCAT:
        return ConcatAggregator()
    if mode is AggregationMode.SEQUENCE:
        return SequenceAggregator()
    raise ValueError(f"Unknown aggregation mode: {mode}")


def _validate_positive(n_units: int, n_steps: int) -> None:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_units <= 0:
        raise ValueError(f"n_units must be positive, got {n_units}")


__all__ = [
    "AggregationMode",
    "StateAggregator",
    "LastAggregator",
    "MeanAggregator",
    "LastMeanAggregator",
    "MTSAggregator",
    "ConcatAggregator",
    "SequenceAggregator",
    "create_aggregator",
]
