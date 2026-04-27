"""
src/reservoir/models/generative.py
Base implementation for generative models providing closed-loop generation.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Protocol, runtime_checkable
from collections.abc import Callable

from beartype import beartype
import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64, TrainLogs, TopologyMeta




@runtime_checkable
class Predictable(Protocol):
    def predict(self, x: JaxF64) -> JaxF64: ...

StateT = TypeVar('StateT')

@beartype
class ClosedLoopGenerativeModel[StateT](ABC):
    """
    Abstract base class for models that can generate autoregressive trajectories.
    Implements the GenerativeModel protocol methods for closed-loop generation.
    """

    topology_meta: TopologyMeta

    def train(self, inputs: JaxF64, targets: JaxF64 | None = None, log_prefix: str = "4", **kwargs) -> TrainLogs:
        """Train the model. Default no-op for models without trainable parameters."""
        return {}

    def get_topology_meta(self) -> TopologyMeta:
        """Return topology metadata dict."""
        return self.topology_meta if hasattr(self, "topology_meta") else {}

    @abstractmethod
    def __call__(self, inputs: JaxF64, return_sequences: bool = False) -> JaxF64:
        """Forward pass: process inputs and return predictions."""
        raise NotImplementedError

    @abstractmethod
    def initialize_state(self, batch_size: int) -> StateT:
        """Initialize hidden state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: StateT, inputs: JaxF64) -> tuple[StateT, JaxF64]:
        """Single time step execution: (state, input) -> (next_state, output)."""
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, state: StateT, input_data: JaxF64) -> tuple[StateT, JaxF64]:
        """Process sequence: (state, input_seq) -> (final_state, output_seq)."""
        raise NotImplementedError

    def generate_closed_loop(
        self,
        seed_data: JaxF64 | None,
        steps: int,
        readout: Predictable | None = None,
        projection_fn: Callable[[JaxF64], JaxF64] | None = None,
        verbose: bool = True,
        initial_state: StateT | None = None,
        initial_output: JaxF64 | None = None,
        return_history: bool = False,
    ) -> JaxF64 | tuple[JaxF64, JaxF64]:
        """
        Generate closed-loop predictions using Fast JAX Scan.
        
        Args:
            seed_data: 2D input (Time, Features) for warmup. Optional if initial_state is provided.
            steps: Number of steps to generate
            readout: Optional readout layer for prediction
            projection_fn: Optional projection function
            verbose: Print progress
            initial_state: Optional pre-computed state to skip warmup.
            initial_output: Optional pre-computed last output to skip warmup.
            return_history: If True, returns (predictions, reservoir_history)
            
        Returns:
            2D predictions (steps, Features) OR (predictions, history)
        """
        
        if initial_state is not None and initial_output is not None:
            # Skip warmup
            final_state = initial_state
            last_output = initial_output
            # Infer batch size from initial_output shape (Batch, Feat)
            batch_size = last_output.shape[0]
        else:
            if seed_data is None:
                raise ValueError("seed_data is required if initial_state/initial_output are not provided.")
                
            # Convert 2D to 3D for internal processing
            history = seed_data
            if history.ndim == 2:
                history = history[None, :, :]  # (T, F) -> (1, T, F)
            elif history.ndim == 1:
                history = history[None, None, :]
            
            batch_size = history.shape[0]
            
            if verbose:
                print(f"[generative.py] Generating {steps} steps (Fast JAX Scan)...")
            
            initial_state_warmup = self.initialize_state(batch_size)
            
            # Apply projection to history if needed
            history_in = projection_fn(history) if projection_fn else history

            if verbose:
                print("[generative.py] Step8  Running forward pass on seed data...")
            final_state, history_outputs = self.forward(initial_state_warmup, history_in)
            last_output = history_outputs[:, -1, :]

        def predict_one(features):
            if readout is not None:
                # Readout expects features in correct shape
                # If features are (batch, feat), readout.predict usually handles it
                # Ensure input is suitable for readout
                f_in = features
                if f_in.ndim == 1:
                    f_in = f_in[None, :]
                
                out = readout.predict(f_in)
                return out
            return features
        
        # Use the last output from history for the first prediction
        first_prediction = predict_one(last_output)
        

        def scan_step(carry, _):
            h_prev, x_raw, step_idx = carry
            
            x_proj = projection_fn(x_raw) if projection_fn else x_raw
            h_next, output = self.step(h_prev, x_proj)
            y_next = predict_one(output)
            
            accum = (y_next, output) if return_history else y_next
            return (h_next, y_next, step_idx + 1), accum
        
        # Initial carry: (state, prediction, step_counter)
        init_carry = (final_state, first_prediction, 0)

        import time
        start_time = time.time()
        print(f"[generative.py] Step 8 jax scan step ...at {time.strftime('%H:%M:%S', time.localtime(start_time))}...")
        _, scan_out = jax.lax.scan(scan_step, init_carry, None, length=steps)
        print(f"[generative.py] Step 8 jax scan step Finished generating in {time.time() - start_time:.4f} seconds.")

        if return_history:
            predictions, history_states = scan_out
            # swap axes: (steps, batch, features) -> (batch, steps, features)
            predictions = jnp.swapaxes(predictions, 0, 1)
            history_states = jnp.swapaxes(history_states, 0, 1)
            
            if batch_size == 1:
                return predictions[0], history_states[0]
            return predictions, history_states
        else:
            predictions = scan_out
            predictions = jnp.swapaxes(predictions, 0, 1)
            if batch_size == 1:
                return predictions[0]
            return predictions
