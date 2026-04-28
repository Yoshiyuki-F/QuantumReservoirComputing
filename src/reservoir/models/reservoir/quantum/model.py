"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/model.py
Quantum Reservoir Model.
Wrapper class that manages state and configuration, delegating computation to functional module.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64, TrainLogs, ConfigDict
from jaxtyping import jaxtyped
from typing import TYPE_CHECKING, Literal, TypedDict
from beartype import beartype

from reservoir.layers.aggregation import AggregationMode
from reservoir.models.reservoir.base import Reservoir, ReservoirConfig
from .backend import _ensure_tensorcircuit_initialized
from .functional import _step_jit, _forward_jit, _get_fused_rotation_matrix

if TYPE_CHECKING:
    from reservoir.layers.projection import Projection

class _QuantumData(TypedDict, total=False):
    """Typed structure for Quantum serialization — avoids ConfigDict union explosion."""
    n_qubits: int
    n_layers: int
    seed: int
    feedback_scale: float
    leak_rate: float
    aggregation: str
    measurement_basis: str
    noise_type: str
    noise_prob: float
    readout_error: float
    n_trajectories: int
    use_remat: bool
    use_reuploading: bool
    precision: str
    chunk_size: int


def _parse_measurement_basis(value: str) -> Literal["Z", "ZZ", "Z+ZZ"]:
    if value == "Z":
        return "Z"
    if value == "ZZ":
        return "ZZ"
    if value == "Z+ZZ":
        return "Z+ZZ"
    raise ValueError(f"Unknown measurement_basis: {value}")


def _parse_noise_type(value: str) -> Literal["clean", "depolarizing", "damping"]:
    if value == "clean":
        return "clean"
    if value == "depolarizing":
        return "depolarizing"
    if value == "damping":
        return "damping"
    raise ValueError(f"Unknown noise_type: {value}")


def _parse_precision(value: str) -> Literal["complex64", "complex128"]:
    if value == "complex64":
        return "complex64"
    if value == "complex128":
        return "complex128"
    raise ValueError(f"Unknown precision: {value}")


def _parse_quantum_data(data: ConfigDict) -> _QuantumData:
    parsed: _QuantumData = {}

    for key in ("n_qubits", "n_layers", "seed", "n_trajectories", "chunk_size"):
        value = data.get(key)
        if value is not None:
            parsed[key] = int(float(str(value)))

    for key in ("feedback_scale", "leak_rate", "noise_prob", "readout_error"):
        value = data.get(key)
        if value is not None:
            parsed[key] = float(str(value))

    for key in ("aggregation", "measurement_basis", "noise_type", "precision"):
        value = data.get(key)
        if value is not None:
            parsed[key] = str(value)

    for key in ("use_remat", "use_reuploading"):
        value = data.get(key)
        if value is not None:
            parsed[key] = bool(value)

    return parsed

class QuantumReservoirConfig(ReservoirConfig):
    n_qubits: int | None
    n_layers: int
    seed: int
    feedback_scale: float
    measurement_basis: Literal["Z", "ZZ", "Z+ZZ"]
    noise_type: Literal["clean", "depolarizing", "damping"]
    noise_prob: float
    readout_error: float
    n_trajectories: int
    use_reuploading: bool
    precision: Literal["complex64", "complex128"]

class QuantumReservoir(Reservoir[tuple[JaxF64, JaxF64 | None]]):
    """
    Quantum Reservoir Computing using TensorCircuit.
    
    - Core Logic (`_step_logic`) is separated from JIT wrappers.
    - `_forward_jit` fuses the scan loop via XLA (no nested JIT).
    - Zero-overhead recompilation (Parameters passed as arguments).
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        seed: int,
        aggregation_mode: AggregationMode,
        feedback_scale: float,     # a_fb: R gate feedback scaling
        leak_rate: float,          # Leaky integrator rate for feedback memory
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
        noise_type: Literal["clean", "depolarizing", "damping"],
        noise_prob: float,
        readout_error: float,
        n_trajectories: int, # 0 means Density Matrix (default), >0 means Monte Carlo
        use_reuploading: bool,

        ## Internal/Advanced Options in JAX (not typically user-facing)
        # TODO should be in some config class or something, not here
        precision: Literal["complex64", "complex128"] = "complex128",
        use_remat: bool = False, # does it matter for forward since we have no trainable params? maybe for step if used standalone?
        chunk_size: int = 32,
    ) -> None:
        """Initialize Quantum Reservoir."""
        # Ensure TC backend and patches are applied (lazy, idempotent)
        _ensure_tensorcircuit_initialized(precision)
        
        n_correlations = n_qubits * (n_qubits - 1) // 2

        # Memory Guardrail
        if n_qubits >= 20:
             import warnings
             warnings.warn(
                 f"n_qubits={n_qubits} requires >100MB per state vector (complex64). "
                 "Ensure you have sufficient VRAM.",
                 ResourceWarning, stacklevel=2
             )
        
        # --- Calculate output dimension based on measurement basis ---
        if measurement_basis == "Z":
            output_dim = n_qubits
        elif measurement_basis == "ZZ":
            output_dim = n_correlations
        elif measurement_basis == "Z+ZZ":
            output_dim = n_qubits + n_correlations
        else:
            output_dim = n_qubits
        
        super().__init__(n_units=output_dim, seed=seed, leak_rate=leak_rate, aggregation_mode=aggregation_mode)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.measurement_basis = measurement_basis
        self.n_correlations = n_correlations
        self.noise_type = noise_type
        self.noise_prob = float(noise_prob)
        self.readout_error = float(readout_error)
        self.n_trajectories = n_trajectories
        self.use_remat = use_remat
        self.use_reuploading = use_reuploading
        self.precision = precision
        self.feedback_scale = float(feedback_scale)  # a_fb: R gate feedback scaling
        self.chunk_size = int(chunk_size)

        self._rng = jax.random.key(seed)
        
        # Initialize Huge Arrays
        self._init_fixed_parameters()
        self._measurement_matrix = self._compute_measurement_matrix_vectorized()

    def _init_fixed_parameters(self) -> None:
        """Initialize fixed random parameters and pre-compute unitaries."""
        # 1. Initialize raw rotation angles (Layer, Qubit, 3)
        raw_params = jax.random.uniform(
            self._rng,
            minval=0.0,
            maxval=2 * jnp.pi,
            shape=(self.n_layers, self.n_qubits, 3)
        )
        
        # 2. Pre-compute fused rotation matrices (Layer, Qubit, 2, 2)
        # We vmap over both layers and qubits
        v_get_matrix = jax.vmap(jax.vmap(_get_fused_rotation_matrix))
        self.reservoir_params = v_get_matrix(raw_params)
        
        self.initial_state_vector = jnp.zeros(self.n_qubits, dtype=jnp.float_)

    def _compute_measurement_matrix_vectorized(self) -> JaxF64:
        """Vectorized Precomputation of Parity/Measurement Matrix."""
        dim = 2 ** self.n_qubits
        basis_states = jnp.arange(dim)
        
        shifts = self.n_qubits - 1 - jnp.arange(self.n_qubits)
        bits = (basis_states[:, None] >> shifts[None, :]) & 1
        z_values = 1.0 - 2.0 * bits
        
        row_blocks = []
        if self.measurement_basis in ("Z", "Z+ZZ"):
            row_blocks.append(z_values.T)
        if self.measurement_basis in ("ZZ", "Z+ZZ"):
            idx_i, idx_j = jnp.triu_indices(self.n_qubits, k=1)
            zz_values = z_values[:, idx_i] * z_values[:, idx_j]
            row_blocks.append(zz_values.T)
            
            
        matrix_np = jnp.vstack(row_blocks)
        
        # Apply Readout Error (Analytical Scaling)
        # Expectation values scale by (1 - 2*epsilon)^weight
        if self.readout_error > 0.0:
            scale_factor = 1.0 - 2.0 * self.readout_error
            # Calculate Hamming weight for each row (observable)
            # Z_i has weight 1. ZZ_ij has weight 2.
            # We constructed rows explicitly:
            # First N rows are Z (weight 1)
            # Next N*(N-1)/2 rows are ZZ (weight 2)
            
            weights = []
            if self.measurement_basis in ("Z", "Z+ZZ"):
                weights.append(self._broadcast_scalar(1, self.n_qubits)) # Weight 1
            if self.measurement_basis in ("ZZ", "Z+ZZ"):
                weights.append(self._broadcast_scalar(2, self.n_correlations)) # Weight 2
                
            w_vec = jnp.concatenate(weights)
            # Reshape for broadcasting: (N_obs, 1)
            damping = (scale_factor ** w_vec)[:, None]
            matrix_np = matrix_np * damping

        return jnp.array(matrix_np)

    @staticmethod
    def _broadcast_scalar(val: int, count: int) -> JaxF64:
        return jnp.full((count,), val)

    def _prepare_input(self, inputs: JaxF64) -> tuple[JaxF64, bool]:
        """Preprocess input: ensure 3D shape (Batch, Time, Feat)."""
        arr = inputs
        input_was_2d = (arr.ndim == 2)
        if input_was_2d:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"QuantumReservoir expects 2D or 3D input, got {arr.shape}")
        return arr, input_was_2d

    def initialize_state(self, batch_size: int = 1) -> tuple[JaxF64, JaxF64 | None]:
        state = jnp.zeros((batch_size, self.n_qubits), dtype=jnp.float_)
        if self.n_trajectories > 0:
            # Monte Carlo Mode: Return (state, key) tuple
            # Update internal RNG state to ensure fresh noise per batch
            key, self._rng = jax.random.split(self._rng)
            return state, jax.random.split(key, batch_size)
        return state, None

    def reset_state(self, batch_size: int) -> tuple[JaxF64, JaxF64 | None]:
        """Alias for initialize_state. Resets the reservoir to the initial ground state."""
        return self.initialize_state(batch_size)

    @jaxtyped(typechecker=beartype)
    def step(self, state: tuple[JaxF64, JaxF64 | None], inputs: JaxF64) -> tuple[tuple[JaxF64, JaxF64 | None], JaxF64]:
        """Batched step function for debugging/stepping."""
        # Use vmapped step logic wrapper
        step_func = partial(
            _step_jit, # Use the JIT compiled wrapper
            reservoir_params=self.reservoir_params,
            measurement_matrix=self._measurement_matrix,
            n_qubits=self.n_qubits,
            feedback_scale=self.feedback_scale,
            leak_rate=self.leak_rate,
            noise_type=self.noise_type,
            noise_prob=self.noise_prob,
            use_remat=self.use_remat,
            use_reuploading=self.use_reuploading
        )
        return jax.vmap(step_func, in_axes=(0, 0))(state, inputs)

    @jaxtyped(typechecker=beartype)
    def forward(self, state: tuple[JaxF64, JaxF64 | None], input_data: JaxF64) -> tuple[tuple[JaxF64, JaxF64 | None], JaxF64]:
        """Forward pass using optimized scan."""
        if input_data.ndim != 3:
            raise ValueError(f"Expected (batch, time, feat), got {input_data.shape}")

        # --- Monte Carlo Ensemble Expansion ---
        original_batch_size = input_data.shape[0]
        traj_count = self.n_trajectories if self.n_trajectories > 1 else 1
        is_ensemble = traj_count > 1
        
        run_state = state
        run_inputs = input_data
        
        if is_ensemble:
            # Polymorphic State Check via Tuple components
            state_vec, rng_keys = state
            if rng_keys is not None:
                # keys shape: (Batch, 2) -> Broaden to (Batch * K, 2)
                
                # Split each key into traj_count keys
                # vmap over batch
                keys_expanded = jax.vmap(lambda k: jax.random.split(k, traj_count))(rng_keys)
                # Shape: (Batch, K, 2) -> Flatten: (Batch*K, 2)
                keys_flat = keys_expanded.reshape(original_batch_size * traj_count, 2)
                
                # Expand Vector: (Batch, N) -> (Batch*K, N)
                state_vec_flat = jnp.repeat(state_vec, traj_count, axis=0)
                
                run_state = (state_vec_flat, keys_flat)
            else:
                # If state is array but n_trajectories > 1 (Maybe density matrix mode overridden?)
                # Just repeat state
                run_state = (jnp.repeat(state_vec, traj_count, axis=0), None)
                
            run_inputs = jnp.repeat(input_data, traj_count, axis=0)
        
        # --- Execution ---
        # Transpose for scan (Time, Batch, Feat)
        inputs_time_major = jnp.swapaxes(run_inputs, 0, 1)
        
        final_state, stacked_outputs = _forward_jit(
            run_state,
            inputs_time_major,
            self.reservoir_params,
            self._measurement_matrix,
            self.n_qubits,
            self.feedback_scale,
            self.leak_rate,
            self.noise_type,
            self.noise_prob,
            self.use_remat,
            self.use_reuploading
        )
        
        # --- Ensemble Aggregation ---
        if is_ensemble:
            # stacked_outputs: (Time, Batch*K, Out) -> Transpose -> (Batch*K, Time, Out)
            stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1) # Now (B*K, T, O)
            
            # Reshape to (Batch, K, Time, Out)
            out_shape = stacked_outputs.shape
            reshaped_out = stacked_outputs.reshape(original_batch_size, traj_count, out_shape[1], out_shape[2])
            
            # Mean over trajectories (axis 1)
            stacked_outputs = jnp.mean(reshaped_out, axis=1) # (Batch, Time, Out)
            
            # Reduce final_state (Optional implementation)
            # Just take the first trajectory's key/state to maintain shape for next steps if needed
            f_vec, f_keys = final_state
            if f_keys is not None:
                 # f_vec: (B*K, N) -> (B, K, N) -> Mean
                 f_vec_mean = f_vec.reshape(original_batch_size, traj_count, -1).mean(axis=1)
                 
                 # f_keys: (B*K, 2). Pick first one per batch.
                 f_keys_reshaped = f_keys.reshape(original_batch_size, traj_count, 2)
                 f_keys_reduced = f_keys_reshaped[:, 0, :]
                 
                 final_state = (f_vec_mean, f_keys_reduced)
            else:
                 f_vec_mean = f_vec.reshape(original_batch_size, traj_count, -1).mean(axis=1)
                 final_state = (f_vec_mean, None)
            
        else:
            # Standard single trajectory / density matrix
            # Transpose back (Batch, Time, Feat)
            stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1)
        
        return final_state, stacked_outputs

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        inputs: JaxF64,
        return_sequences: bool = False,
        split_name: str | None = None,
    ) -> JaxF64:
        arr, input_was_2d = self._prepare_input(inputs)
        
        batch_size = arr.shape[0]
        state = self.initialize_state(batch_size)
        final_state, states = self.forward(state, arr)
        
        if return_sequences:
            return states[0] if input_was_2d else states
        
        log_label = f"6:{split_name}" if split_name else None
        return self.aggregator.transform(states, log_label=log_label)



    def train(
        self,
        inputs: JaxF64,
        targets: JaxF64 | None = None,
        log_prefix: str = "4",
        projection_layer: Projection | None = None,
    ) -> TrainLogs:
        # Reservoir has no trainable parameters; arguments are unused.
        return {}

    def to_dict(self) -> ConfigDict:
        base_data = super().to_dict()
        return {
            "n_units": base_data["n_units"],
            "leak_rate": base_data["leak_rate"],
            "aggregation": base_data["aggregation"],
            "n_qubits": int(self.n_qubits) if self.n_qubits is not None else None,
            "n_layers": int(self.n_layers),
            "seed": int(self.seed),
            "feedback_scale": float(self.feedback_scale),
            "measurement_basis": str(self.measurement_basis),
            "noise_type": str(self.noise_type),
            "noise_prob": float(self.noise_prob),
            "readout_error": float(self.readout_error),
            "n_trajectories": int(self.n_trajectories),
            "use_reuploading": bool(self.use_reuploading),
            "precision": str(self.precision),
            "use_remat": bool(self.use_remat),
            "chunk_size": int(self.chunk_size),
        }

    @classmethod
    def from_dict(cls, data: ConfigDict) -> QuantumReservoir:
        d = _parse_quantum_data(data)
        try:
            return cls(
                n_qubits=int(d["n_qubits"]),
                n_layers=int(d["n_layers"]),
                seed=int(d["seed"]),
                feedback_scale=float(d.get("feedback_scale", 0.0)),
                leak_rate=float(d.get("leak_rate", 1.0)),
                aggregation_mode=AggregationMode(str(d["aggregation"])),
                measurement_basis=_parse_measurement_basis(d.get("measurement_basis", "Z")),
                noise_type=_parse_noise_type(d.get("noise_type", "clean")),
                noise_prob=float(d.get("noise_prob", 0.0)),
                readout_error=float(d.get("readout_error", 0.0)),
                n_trajectories=int(d.get("n_trajectories", 0)),
                use_reuploading=bool(d.get("use_reuploading", False)),
                precision=_parse_precision(d.get("precision", "complex64")),
                use_remat=bool(d.get("use_remat", False)),
                chunk_size=int(d.get("chunk_size", 32)),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required quantum reservoir parameter '{exc.args[0]}'") from exc

    def get_observable_names(self) -> list[str]:
        """Generate human-readable names for the measured observables."""
        names = []
        if self.measurement_basis in ("Z", "Z+ZZ"):
            names.extend([f"Z{i}" for i in range(self.n_qubits)])
        
        if self.measurement_basis in ("ZZ", "Z+ZZ"):
            # Same order as _compute_measurement_matrix_vectorized: triu indices
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    names.append(f"Z{i}Z{j}")
        return names

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(tc_backend, n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"lr{self.leak_rate:.4f}_f{self.feedback_scale:.4f}, "
            f"measurement={self.measurement_basis}, "
            f"noise={self.noise_type}({self.noise_prob}), ro_err={self.readout_error}, mc={self.n_trajectories}, "
            f"remat={self.use_remat}, reup={self.use_reuploading}, {self.precision})"
        )
