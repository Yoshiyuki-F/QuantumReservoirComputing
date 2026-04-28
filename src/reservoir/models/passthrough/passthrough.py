"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/passthrough/passthrough.py
Step 5 SKIP: Passthrough model that applies only aggregation (Step 6) to projected input.

Flow: [Batch, Time, Hidden] -> Aggregation -> [Batch, Feature]
"""
from __future__ import annotations


from beartype import beartype
import jax
import jax.numpy as jnp

from reservoir.layers.aggregation import AggregationMode
from reservoir.models.generative import ClosedLoopGenerativeModel
from reservoir.layers.aggregation import create_aggregator

from reservoir.core.types import JaxF64, TrainLogs, TopologyMeta  # noqa: TC001
from reservoir.layers.projection import Projection  # noqa: TC001



@beartype
class PassthroughModel(ClosedLoopGenerativeModel[JaxF64]):
    """
    Model that skips dynamics (Step 5) and directly aggregates projected features.
    Implements the same interface as Reservoir for compatibility with GenericRunner.
    """

    def __init__(self, aggregation_mode: AggregationMode) -> None:
        if not isinstance(aggregation_mode, AggregationMode):
            raise TypeError(f"aggregation_mode must be AggregationMode, got {type(aggregation_mode)}.")
        self.aggregator = create_aggregator(aggregation_mode)
        self.topology_meta: TopologyMeta = {}
        self._n_units: int | None = None  # Set on first forward pass

    def train(
        self,
        inputs: JaxF64,
        targets: JaxF64 | None = None,
        log_prefix: str = "4",
        projection_layer: Projection | None = None,
    ) -> TrainLogs:
        """No-op: Passthrough has no trainable parameters."""
        return {}

    # ------------------------------------------------------------------ #
    # Stateful Interface (Compatible with Reservoir)                     #
    # ------------------------------------------------------------------ #

    def initialize_state(self, batch_size: int) -> JaxF64:
        """Return zero state. Passthrough is stateless but needs compatible interface."""
        if self._n_units is None:
            raise RuntimeError("Passthrough n_units not set. Call forward() first.")
        return jnp.zeros((batch_size, self._n_units))

    def step(self, state: JaxF64, inputs: JaxF64) -> tuple[JaxF64, JaxF64]:
        """Passthrough step: ignore state, return input as next state."""
        # inputs: [batch, features] - ensure dtype matches state
        next_state = inputs
        return next_state, next_state

    def forward(self, state: JaxF64, input_data: JaxF64) -> tuple[JaxF64, JaxF64]:
        """Process sequence. Returns (final_state, all_states)."""
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, input), got {input_data.shape}")
        
        # Use scan for consistency with Reservoir
        proj_transposed = jnp.swapaxes(input_data, 0, 1)  # [time, batch, feat]
        final_states, stacked = jax.lax.scan(self.step, state, proj_transposed)
        stacked = jnp.swapaxes(stacked, 0, 1)  # [batch, time, feat]
        return final_states, stacked
    
    # generate_closed_loop is inherited from ClosedLoopGenerativeModel

    # ------------------------------------------------------------------ #
    # Standard Interface                                                 #
    # ------------------------------------------------------------------ #

    def __call__(self, inputs: JaxF64, return_sequences: bool = False, split_name: str | None = None) -> JaxF64:

        """Aggregate projected features. Accepts both 2D (Time, Features) and 3D (Batch, Time, Features). Output is 2D."""
        arr = inputs
        
        # Convert 2D to 3D for internal processing
        if arr.ndim == 2:
            arr = arr[None, :, :]  # (T, F) -> (1, T, F)
        elif arr.ndim != 3:
            raise ValueError(f"PassthroughModel expects 2D or 3D input, got {arr.shape}")
        
        # Track n_units for initialize_state
        self._n_units = arr.shape[-1]
        
        # Aggregation always returns 2D
        log_label = f"6:{split_name}" if split_name else None
        return self.aggregator.transform(arr, log_label=log_label)

    def get_feature_dim(self, n_units: int, time_steps: int) -> int:
        """Return aggregated feature dimension."""
        return self.aggregator.get_output_dim(n_units, int(time_steps))

    def get_topology_meta(self) -> TopologyMeta:
        return self.topology_meta

    def __repr__(self) -> str:
        return f"PassthroughModel(agg={self.aggregator.mode.value})"
