"""
src/reservoir/models/reservoir/base.py
Base class for Reservoir Computing models implementing ReservoirNode protocol.
"""
from abc import ABC, abstractmethod
from beartype import beartype
from reservoir.core.types import JaxF64, ConfigDict, KwargsDict, TopologyMeta
from typing import TypedDict, TypeVar

from reservoir.layers.aggregation import AggregationMode
from reservoir.layers.aggregation import create_aggregator
from reservoir.models.generative import ClosedLoopGenerativeModel

StateT = TypeVar("StateT", default=JaxF64)

class ReservoirConfig(TypedDict):
    n_units: int
    leak_rate: float
    aggregation: str

@beartype
class Reservoir[StateT](ClosedLoopGenerativeModel[StateT], ABC):
    """Abstract base class providing common scan-based trajectory generation."""

    def __init__(self, n_units: int, seed: int, leak_rate: float, aggregation_mode: AggregationMode) -> None:
        self.n_units = n_units
        self.seed = seed
        self.leak_rate = float(leak_rate)
        
        if not (0.0 < self.leak_rate <= 1.0):
             raise ValueError(f"leak_rate must be in (0,1], got {self.leak_rate}")

        if not isinstance(aggregation_mode, AggregationMode):
            raise TypeError(f"aggregation_mode must be AggregationMode, got {aggregation_mode.__class__.__name__}.")
            
        self.aggregator = create_aggregator(aggregation_mode)

    @property
    def output_dim(self) -> int:
        return self.n_units

    @abstractmethod
    def initialize_state(self, batch_size: int = 1) -> StateT:
        """Initialize reservoir state (must be implemented by subclasses)."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: StateT, input_data: JaxF64) -> tuple[StateT, JaxF64]:
        """Compute single step dynamics returning next state and emitted features."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: StateT, inputs: JaxF64) -> tuple[StateT, JaxF64]:
        """Single time step - used for closed-loop generation."""
        raise NotImplementedError

    # generate_closed_loop is inherited from ClosedLoopGenerativeModel

    def generate_trajectory(self, initial_state: StateT, inputs: JaxF64) -> JaxF64:

        """Process sequences by delegating to the subclass forward implementation."""
        is_sequence_batched = inputs.ndim == 3
        inputs_batched = inputs if is_sequence_batched else inputs[None, ...]
        
        # State batch handling depends on whether the state is a tuple or a direct array
        # This allows polymorphism from children implementing Generic[StateT]
        
        # Assume initial_state is correctly shaped or typed (Trust the Type System)
        init_batched = initial_state

        _, outputs = self.forward(init_batched, inputs_batched)
        states = outputs.states if hasattr(outputs, "states") else outputs
        return states if is_sequence_batched else states[0]

    def to_dict(self) -> ConfigDict:
        return {
            "n_units": self.n_units,
            "leak_rate": self.leak_rate,
            "aggregation": self.aggregator.mode.value,
        }

    def get_topology_meta(self) -> TopologyMeta:
        """Optional topology metadata set by factories."""
        return self.topology_meta

    def get_feature_dim(self, time_steps: int) -> int:
        """Return aggregated feature dimension without running the model."""
        return self.aggregator.get_output_dim(self.n_units, int(time_steps))

    def __call__(self, inputs: JaxF64, params: KwargsDict | None = None) -> JaxF64:
        """
        Allow reservoir nodes to be used directly in SequentialModel.
        Automatically initializes state and runs trajectory generation.
        """
        arr = inputs
        if arr.ndim not in (2, 3):
            raise ValueError(f"Reservoir input must be 2D or 3D, got {arr.shape}")
        batch_size = arr.shape[0] if arr.ndim == 3 else 1
        init_state = self.initialize_state(batch_size)
        return self.generate_trajectory(init_state, arr)


type ReservoirModel = Reservoir[JaxF64] | Reservoir[tuple[JaxF64, JaxF64 | None]]
