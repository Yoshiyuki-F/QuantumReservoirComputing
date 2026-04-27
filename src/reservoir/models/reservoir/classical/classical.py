"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/classical.py
Standard Echo State Network implementation.
"""

from __future__ import annotations


from beartype import beartype
import jax
import jax.numpy as jnp
import typing

from reservoir.layers.aggregation import AggregationMode
from reservoir.models.reservoir.base import Reservoir

from reservoir.core.types import JaxF64, TrainLogs, ConfigDict



@beartype
class ClassicalReservoir(Reservoir):
    """Minimal ESN-style reservoir built on the Reservoir base class."""

    def __init__(
        self,
        n_units: int,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        seed: int,
        aggregation_mode: AggregationMode,
    ) -> None:
        super().__init__(n_units=n_units, seed=seed, leak_rate=leak_rate, aggregation_mode=aggregation_mode)
        self.spectral_radius = float(spectral_radius)
        self.rc_connectivity = float(rc_connectivity)
        self._rng = jax.random.PRNGKey(self.seed)
        self._init_weights()

    def _init_weights(self) -> None:
        k_res, k_mask_res = jax.random.split(self._rng, 2)
        self._rng = k_res

        W_dense = jax.random.normal(k_res, (self.n_units, self.n_units))
        if 0.0 < self.rc_connectivity < 1.0:
            mask_res = jax.random.bernoulli(k_mask_res, p=self.rc_connectivity, shape=W_dense.shape)
            W_dense = jnp.where(mask_res, W_dense, 0.0)

        eig = jnp.max(jnp.abs(jnp.linalg.eigvals(W_dense)))
        scale = self.spectral_radius / eig if eig > 0 else 1.0
        self.W = W_dense * scale

    def initialize_state(self, batch_size: int = 1) -> JaxF64:
        return jnp.zeros((batch_size, self.n_units))

    def step(self, state: JaxF64, inputs: JaxF64) -> tuple[JaxF64, JaxF64]:

        # next_state = (1.0 - self.leak_rate) * state + jnp.tanh(inputs + jnp.dot(state, self.W))
        # Jaeger (2007) Standard Li-ESN
        next_state = (1.0 - self.leak_rate) * state + self.leak_rate * jnp.tanh(inputs + jnp.dot(state, self.W))

        return next_state, next_state

    def forward(self, state: JaxF64, input_data: JaxF64) -> tuple[JaxF64, JaxF64]:
        if input_data.ndim != 3:
            raise ValueError(f"Expected batched sequences (batch, time, input), got {input_data.shape}")
        batch, time, feat = input_data.shape
        if feat != self.n_units:
            raise ValueError(f"Reservoir expects feature dim {self.n_units}, got {feat}")
        projected = input_data
        proj_transposed = jnp.swapaxes(projected, 0, 1)

        final_states, stacked = jax.lax.scan(self.step, state, proj_transposed)
        stacked = jnp.swapaxes(stacked, 0, 1)
        return final_states, stacked

    def __call__(self, inputs: JaxF64, return_sequences: bool = False, split_name: str | None = None, **kwargs: typing.Any) -> JaxF64:
        """Process inputs. Accepts both 2D (Time, Features) and 3D (Batch, Time, Features). Output is 2D."""
        arr = inputs
        
        # Convert 2D to 3D for internal processing (scan requires 3D)
        input_was_2d = (arr.ndim == 2)
        if input_was_2d:
            arr = arr[None, :, :]  # (T, F) -> (1, T, F)
        elif arr.ndim != 3:
            raise ValueError(f"ClassicalReservoir expects 2D or 3D input, got {arr.shape}")
        
        batch_size = arr.shape[0]
        initial_state = self.initialize_state(batch_size)
        final_state, artifacts = self.forward(initial_state, arr)
        states = artifacts

        if return_sequences:
            return states[0] if input_was_2d else states
        
        # Aggregation always returns 2D
        log_label = f"6:{split_name}" if split_name else None
        return self.aggregator.transform(states, log_label=log_label)



    def train(self, inputs: JaxF64, targets: JaxF64 | None = None, log_prefix: str = "4", **kwargs) -> TrainLogs:
        """
        Reservoir has no trainable parameters; run forward for compatibility and return empty logs.
        """
        return {}

    def to_dict(self) -> ConfigDict:
        data = super().to_dict()
        res: ConfigDict = dict(data)
        res.update(
            {
                "spectral_radius": self.spectral_radius,
                "rc_connectivity": self.rc_connectivity,
            }
        )
        return res

    @classmethod
    def from_dict(cls, data: ConfigDict) -> ClassicalReservoir:
        try:
            return cls(
                n_units=int(float(str(data["n_units"]))),
                spectral_radius=float(str(data["spectral_radius"])),
                leak_rate=float(str(data["leak_rate"])),
                rc_connectivity=float(str(data["rc_connectivity"])),
                seed=int(float(str(data["seed"]))),
                aggregation_mode=AggregationMode(str(data["aggregation"])),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required reservoir parameter '{exc.args[0]}'") from exc

    def __repr__(self) -> str:
        return f"ClassicalReservoir(n_units={self.n_units})"
