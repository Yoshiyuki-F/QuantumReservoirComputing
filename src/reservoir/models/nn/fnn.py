"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/fnn.py
FNN BaseModel wrapper using BaseFlaxModel.
Adapter is selected based on FNNConfig: Flatten (default) or TimeDelayEmbedding (if window_size is set).
"""

from __future__ import annotations

from collections.abc import Sequence, Callable  # noqa: TC003
from reservoir.training.presets import TrainingConfig  # noqa: TC001
from reservoir.core.types import JaxF64, TrainLogs, EvalMetrics  # noqa: TC001
from reservoir.layers.projection import Projection  # noqa: TC001

from beartype import beartype
import flax.linen as nn
import jax.numpy as jnp

from reservoir.models.config import FNNConfig
from reservoir.models.nn.base import BaseFlaxModel
from reservoir.models.generative import ClosedLoopGenerativeModel, Predictable  # noqa: TC001
from reservoir.layers.adapters import Adapter, Flatten, TimeDelayEmbedding


# 全体の実行を通して、モデルの構造(Shape)を一度だけ出力するためのフラグ
_LOGGED_ONCE = False


@beartype
class FNNModel(BaseFlaxModel, ClosedLoopGenerativeModel[JaxF64]):
    """
    FNN with configurable adapter (Step 4).
    
    If config.window_size is None: uses Flatten adapter (for classification)
    If config.window_size is set: uses TimeDelayEmbedding adapter (for time series regression)
    
    Also implements ClosedLoopGenerativeModel for autoregressive generation.
    """

    @property
    def input_window_size(self) -> int:
        return self.window_size or 0

    def __init__(self, model_config: FNNConfig, input_dim: int, output_dim: int, classification: bool, training_config: TrainingConfig | None = None):
        if not isinstance(model_config, FNNConfig):
            raise TypeError(f"FNNModel expects FNNConfig, got {model_config.__class__.__name__}.")
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("input_dim and output_dim must be positive for FNNModel.")

        # Store window configuration
        self.window_size = model_config.window_size
        self._output_dim = output_dim
        
        # Select adapter based on config
        self.adapter: Adapter
        if self.window_size is not None:
            # TimeDelayEmbedding: adapter for windowed time series
            self.adapter = TimeDelayEmbedding(window_size=self.window_size)
        else:
            # Flatten: standard behavior for classification or global sequences
            self.adapter = Flatten()

        # input_dim passed to the constructor is now the EFFECTIVE dimension (post-adapter)
        # e.g., 784 for MNIST (28*28), or window_size * features for TDE.
        effective_input_dim = int(input_dim)

        hidden_layers = tuple(int(h) for h in (model_config.hidden_layers or ()))
        hidden_layers = tuple(h for h in hidden_layers if h > 0)

        self.layer_dims: tuple[int, ...] = (effective_input_dim, *hidden_layers, int(output_dim))

        super().__init__({"layer_dims": self.layer_dims}, classification=classification, training_config=training_config)

    def train(
        self,
        inputs: JaxF64,
        targets: JaxF64 | None = None,
        log_prefix: str = "4",
        projection_layer: Projection | None = None,
        adapter: Adapter | None = None,
    ) -> TrainLogs:
        """Train with optional on-the-fly projection and adaptation inside the batched loop."""
        # We pass raw inputs directly to base class which will handle batching, projection, and adaptation
        return super().train(
            inputs,
            targets,
            log_prefix=log_prefix,
            projection_layer=projection_layer,
            adapter=self.adapter,
        )

    def predict(self, X: JaxF64, projection_layer: Projection | None = None) -> JaxF64:
        """Predict with optional dynamic projection and adapter-transformed inputs."""
        if projection_layer is not None:
             X = projection_layer(X)
             
        # Check if inputs are already adapted
        if X.ndim == 2 and X.shape[-1] == self.layer_dims[0]:
            return super().predict(X)
            
        adapted_inputs = self.adapter(X)
        return super().predict(adapted_inputs)

    def evaluate(self, X: JaxF64, y: JaxF64) -> EvalMetrics:
        """Evaluate with adapter-transformed inputs (and aligned targets if windowed)."""
        # Check if inputs are already adapted
        if X.ndim == 2 and X.shape[-1] == self.layer_dims[0]:
            return super().evaluate(X, y)

        adapted_inputs = self.adapter(X)
        aligned_targets = self.adapter.align_targets(y)
        return super().evaluate(adapted_inputs, aligned_targets)

    def __call__(self, inputs: JaxF64, return_sequences: bool = False) -> JaxF64:
        """Make model callable for batched_compute compatibility."""
        return self.predict(inputs)

    def _create_model_def(self) -> nn.Module:
        return FNN(layer_dims=self.layer_dims, return_hidden=False)

    # ------------------------------------------------------------------ #
    # ClosedLoopGenerativeModel implementation                           #
    # ------------------------------------------------------------------ #
    def initialize_state(self, batch_size: int = 1) -> JaxF64:
        """
        For windowed FNN, state is the sliding window buffer of last `window_size` values.
        For non-windowed FNN, state is a dummy.
        """
        if self.window_size is not None:
            # State holds the sliding window buffer: (batch, window_size, features)
            return jnp.zeros((batch_size, self.window_size, self._output_dim))
        return jnp.zeros((batch_size, 1))

    def step(self, state: JaxF64, inputs: JaxF64) -> tuple[JaxF64, JaxF64]:
        """
        Single step for closed-loop generation.
        For windowed FNN: concatenate window buffer to form input, predict, update buffer.
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        
        if self.window_size is not None:
            # state shape: (batch, window_size, features)
            # Flatten window to (batch, window_size * features) for FNN input
            batch_size = state.shape[0]
            windowed_input = state.reshape(batch_size, -1)  # (batch, window_size * features)
            
            # Predict using base class (no adapter, already windowed)
            output = super().predict(windowed_input)  # (batch, output_dim)
            
            # Update sliding window: shift left and append new prediction
            # new_state = [state[:, 1:, :], output[:, None, :]]
            new_state = jnp.concatenate([state[:, 1:, :], output[:, None, :]], axis=1)
            
            return new_state, output
        else:
            # Non-windowed: simple pass-through
            output = super().predict(inputs)
            return state, output

    def forward(self, state: JaxF64, input_data: JaxF64) -> tuple[JaxF64, JaxF64]:
        """
        Process seed sequence to initialize the sliding window state.
        For windowed FNN: fill the window buffer with the last `window_size` values from input.
        """
        # input_data shape: (batch, time, features)
        batch_size, seq_len, feat_dim = input_data.shape
        
        if self.window_size is not None:
            # Initialize state with last window_size values from seed
            # State shape: (batch, window_size, features)
            if seq_len >= self.window_size:
                initial_window = input_data[:, -self.window_size:, :]
            else:
                # Pad with zeros if seed is shorter than window
                padding = jnp.zeros((batch_size, self.window_size - seq_len, feat_dim))
                initial_window = jnp.concatenate([padding, input_data], axis=1)
            
            # Process seed through the windowed FNN to get outputs
            outputs: list[JaxF64] = []
            for t in range(self.window_size - 1, seq_len):
                # Build window from input_data
                window = input_data[:, t - self.window_size + 1:t + 1, :]  # (batch, window_size, feat)
                windowed_input = window.reshape(batch_size, -1)
                output = super().predict(windowed_input)
                outputs.append(output)
            
            # Return final state (last window) and outputs
            output_seq = jnp.stack(outputs, axis=1) if outputs else jnp.zeros((batch_size, 0, self._output_dim))
            return initial_window, output_seq
        else:
            # Non-windowed: predict each timestep
            outputs: list[JaxF64] = []
            for t in range(seq_len):
                step_input = input_data[:, t, :]
                output = super().predict(step_input)
                outputs.append(output)
            output_seq = jnp.stack(outputs, axis=1)
            return state, output_seq

    def generate_closed_loop(
        self,
        seed_data: JaxF64 | None,
        steps: int,
        readout: Predictable | None = None,
        projection_fn: Callable[[JaxF64], JaxF64] | None = None,
        verbose: bool = True,
        initial_state: JaxF64 | None = None,
        initial_output: JaxF64 | None = None,
        return_history: bool = False,
    ) -> JaxF64 | tuple[JaxF64, JaxF64]:
        """
        FNN-specific closed-loop generation using simple 2D arrays.
        Optimized with jax.lax.scan for speed.
        """
        import jax

        if seed_data is None:
            raise ValueError("FNNModel.generate_closed_loop requires seed_data.")
        
        if self.window_size is None:
            # Non-windowed: just predict directly
            predictions = self.predict(seed_data)
            return (predictions, predictions) if return_history else predictions
        
        # seed_data: (time, features) - use last window_size values as initial state
        seed = seed_data
        if seed.ndim == 3:
            seed = seed[0]  # Remove batch dim if present: (1, T, F) -> (T, F)
        
        W = self.window_size
        feat_dim = seed.shape[-1]
        
        # Initial window buffer: last W values from seed
        if seed.shape[0] >= W:
            window = seed[-W:, :]  # (W, features)
        else:
            padding = jnp.zeros((W - seed.shape[0], feat_dim))
            window = jnp.concatenate([padding, seed], axis=0)
        
        if verbose:
            print(f"    [FNN] Generating {steps} steps (JAX scan, window={W})...")
        
        def scan_step(window_state: JaxF64, _: None) -> tuple[JaxF64, JaxF64]:
            # window_state: (W, features)
            # Flatten to (1, W * features) for FNN
            windowed_input = window_state.reshape(1, -1)
            output = super(FNNModel, self).predict(windowed_input)  # (1, output_dim)
            output = output.squeeze(0)  # (output_dim,)
            
            # Shift window: drop oldest, append new prediction
            new_window = jnp.concatenate([window_state[1:, :], output[None, :]], axis=0)
            return new_window, output
        
        _, predictions = jax.lax.scan(scan_step, window, None, length=steps)
        
        # predictions: (steps, output_dim)
        return (predictions, predictions) if return_history else predictions


class FNN(nn.Module):
    """Feed-forward network whose depth/width comes from layer_dims."""

    layer_dims: Sequence[int] = ()
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: JaxF64):
        global _LOGGED_ONCE
        # assume x is already jax array from batched_compute
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions")

        # JAXのコンパイル（トレース）時に1回だけ呼ばれる通常のprintを使うことで、
        # 自動的に「最初の1回（コンパイル時）」だけShape等が出力されます。
        # _LOGGED_ONCEフラグを使って、訓練(batch>1)の最初の1バッチ/Shapeのみを出力します。
        is_training_batch = x.shape[0] > 1
        should_log = is_training_batch and not _LOGGED_ONCE
        
        if should_log:
            print(f"FNN Input Shape (Batchsize, Features) = {x.shape}")
            _LOGGED_ONCE = True

        hidden_output = None
        target_dims = self.layer_dims[1:]

        for idx, feat in enumerate(target_dims):
            x = nn.Dense(features=feat, dtype=jnp.float64, param_dtype=jnp.float64)(x)

            if should_log:
                print(f"Layer {idx} output shape (Batchsize, Features) = {x.shape}")

            is_last = idx == len(target_dims) - 1
            if not is_last:
                x = nn.relu(x)
                hidden_output = x

        if hidden_output is None:
            hidden_output = x

        if self.return_hidden:
            return x, hidden_output
        return x
