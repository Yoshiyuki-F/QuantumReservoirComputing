"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/reservoir/quantum/functional.py
Pure functional implementation of Quantum Reservoir logic.
Final High-Performance Engine: Static Pre-computation + Brickwork Parallelization.
Optimized for complex128 precision.
"""

from __future__ import annotations

from typing import cast, TYPE_CHECKING
from functools import partial, lru_cache

import jax
import jax.numpy as jnp
import tensorcircuit as tc

from .backend import I_MAT, X_MAT, Y_MAT, Z_MAT

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, JaxKey


# --- Pre-computed Constants ---


@lru_cache(maxsize=8)
def _get_qubit_offsets(n_qubits: int) -> JaxF64:
    """Pre-computed per-qubit angular offsets for input diversification.
    Cached to avoid redundant jnp.linspace calls inside JIT.
    """
    return jnp.linspace(0.0, jnp.pi, n_qubits, endpoint=False)

# --- used in model.py

def _get_fused_rotation_matrix(params: JaxF64) -> JaxF64:
    """Fuse RX, RY, RZ rotations into a single 2x2 unitary matrix."""
    tx, ty, tz = params[0], params[1], params[2]

    def rot_x(p: JaxF64) -> JaxF64:
        c = jnp.cos(p / 2)
        s = -1.0j * jnp.sin(p / 2)
        return jnp.stack(
            [jnp.stack([c, s], axis=-1), jnp.stack([s, c], axis=-1)], axis=-2
        )

    def rot_y(p: JaxF64) -> JaxF64:
        c = jnp.cos(p / 2)
        s = jnp.sin(p / 2)
        return jnp.stack(
            [jnp.stack([c, -s], axis=-1), jnp.stack([s, c], axis=-1)], axis=-2
        )

    def rot_z(p: JaxF64) -> JaxF64:
        c = jnp.cos(p / 2)
        s = -1.0j * jnp.sin(p / 2)
        en = c + s
        ep = c - s
        z = jnp.zeros_like(p, dtype=jnp.complex128)
        return jnp.stack(
            [jnp.stack([en, z], axis=-1), jnp.stack([z, ep], axis=-1)], axis=-2
        )

    return jnp.matmul(rot_z(tz), jnp.matmul(rot_y(ty), rot_x(tx)))

# --- Performance Optimization Helpers ---

def _get_paper_R_unitary(theta: JaxF64) -> JaxF64:
    """Compute fused 4x4 unitary for the Paper R gate (vmap safe).
    Analytically pre-computed to avoid kron and matmul overhead.
    """
    c = jnp.cos(theta / 2.0)
    s = -1.0j * jnp.sin(theta / 2.0)
    zm = c + s
    zp = c - s

    c2 = c * c
    s2 = s * s
    cs = c * s

    return jnp.stack(
        [
            jnp.stack([zm * c2, zm * cs, zm * cs, zm * s2], axis=-1),
            jnp.stack([zp * cs, zp * c2, zp * s2, zp * cs], axis=-1),
            jnp.stack([zp * cs, zp * s2, zp * c2, zp * cs], axis=-1),
            jnp.stack([zm * s2, zm * cs, zm * cs, zm * c2], axis=-1),
        ],
        axis=-2,
    )

def _get_input_unitaries(u_vec: JaxF64, n_qubits: int) -> JaxF64:
    """Computes input unitaries with per-qubit diversification offsets."""
    indices = jnp.arange(n_qubits) % u_vec.shape[0]
    base_angles = u_vec[indices]
    offsets = _get_qubit_offsets(n_qubits)
    return jax.vmap(_get_paper_R_unitary)(base_angles + offsets)


def _get_fused_input_and_feedback(input_unitaries: JaxF64, feedback_val: JaxF64, feedback_scale: float) -> JaxF64:
    """Computes feedback unitaries and fuses them with the input unitaries."""
    fb_unitaries = jax.vmap(_get_paper_R_unitary)(feedback_val * feedback_scale)

    # SWAP trick: align feedback qubit ordering (i+1, i) → (i, i+1) before fusing.
    # This is essentially free in JAX (index remap, no FLOPs).
    swap_idx = jnp.array([0, 2, 1, 3])
    fb_swapped = fb_unitaries[:, swap_idx, :][:, :, swap_idx]
    return jnp.matmul(fb_swapped, input_unitaries)



# --- Core Step Logic ---


def _make_circuit_logic(
    input_unitaries: JaxF64,  # (N, 4, 4)
    feedback_val: JaxF64,
    params: JaxF64,  # (L, N, 2, 2)
    n_qubits: int,
    feedback_scale: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
    rng_key: JaxKey | None = None,
) -> JaxF64:
    tc.set_backend("jax")
    is_noisy = noise_type != "clean"
    is_mc = rng_key is not None

    fused_unitaries = _get_fused_input_and_feedback(input_unitaries, feedback_val, feedback_scale)

    c = tc.Circuit(n_qubits)
    # Encoding: Fused Input + Feedback (applied to same pairs)
    for q_i in range(n_qubits):
        c.unitary(q_i, (q_i + 1) % n_qubits, unitary=fused_unitaries[q_i])  # type: ignore[bad-argument-type]

    state = c.state()
    if is_noisy and state.ndim == 1:
        state = jnp.outer(state, jnp.conj(state))

    def layer_step(
        carry_state: JaxF64 | tuple[JaxF64, JaxKey | None],
        layer_rotation_unitaries: JaxF64,
    ) -> tuple[JaxF64 | tuple[JaxF64, JaxKey | None], None]:
        current_key = None
        if is_mc:
            state_vec, current_key = carry_state
            lc = tc.Circuit(n_qubits, inputs=state_vec)
        elif is_noisy:
            lc = tc.DMCircuit(n_qubits, inputs=carry_state)
        else:
            lc = tc.Circuit(n_qubits, inputs=carry_state)

        def apply_noise(noise_indices: list[int]) -> None:
            nonlocal current_key
            if not is_noisy:
                return
            
            r_array: JaxF64 | None = None
            if is_mc:
                if current_key is None:
                    raise RuntimeError("current_key must be initialized for MC noise")
                # Vectorized PRNG split: generate all random numbers for this batch of indices at once
                k1, current_key = jax.random.split(cast("jax.Array", current_key))
                r_array = jax.random.uniform(k1, shape=(len(noise_indices),))
                
            for enum_i, target_idx in enumerate(noise_indices):
                if is_mc:
                    if r_array is None:
                        raise RuntimeError("r_array must be initialized for MC noise")
                    r = r_array[enum_i]
                    gate_idx = jnp.zeros((), dtype=jnp.int32)
                    gate_idx = jax.lax.select(
                        jnp.array(r < 3 * noise_prob, dtype=bool), jnp.array(3), gate_idx
                    )
                    gate_idx = jax.lax.select(
                        jnp.array(r < 2 * noise_prob, dtype=bool), jnp.array(2), gate_idx
                    )
                    gate_idx = jax.lax.select(jnp.array(r < noise_prob, dtype=bool), jnp.array(1), gate_idx)
                    mat = jax.lax.switch(
                        gate_idx,
                        [
                            lambda _: I_MAT,
                            lambda _: X_MAT,
                            lambda _: Y_MAT,
                            lambda _: Z_MAT,
                        ],
                        None,
                    )
                    lc.unitary(target_idx, unitary=mat)  # type: ignore[bad-argument-type]
                else:
                    if noise_type == "depolarizing":
                        lc.depolarizing(  # type: ignore[bad-argument-type]
                            target_idx, px=noise_prob, py=noise_prob, pz=noise_prob
                        )
                    elif noise_type == "damping":
                        lc.amplitude_damping(target_idx, gamma=noise_prob)  # type: ignore[bad-argument-type]

        if use_reuploading:
            for enc_q_idx in range(n_qubits):
                lc.unitary(enc_q_idx, (enc_q_idx + 1) % n_qubits, unitary=input_unitaries[enc_q_idx])  # type: ignore[bad-argument-type]
            for noise_q_idx in range(n_qubits):
                apply_noise([noise_q_idx])

        # Brickwork Entanglement
        for j in range(0, n_qubits - 1, 2):
            lc.cnot(j, j + 1)  # type: ignore[bad-argument-type]
            apply_noise([j, j + 1])
        for j in range(1, n_qubits - 1, 2):
            lc.cnot(j, j + 1)  # type: ignore[bad-argument-type]
            apply_noise([j, j + 1])
        for k in range(n_qubits):
            lc.unitary(k, unitary=layer_rotation_unitaries[k])  # type: ignore[bad-argument-type]
            apply_noise([k])
        for j in range(1, n_qubits - 1, 2):
            lc.cnot(j, j + 1)  # type: ignore[bad-argument-type]
            apply_noise([j, j + 1])
        for j in range(0, n_qubits - 1, 2):
            lc.cnot(j, j + 1)  # type: ignore[bad-argument-type]
            apply_noise([j, j + 1])

        new_state = lc.state()
        return (new_state, current_key) if is_mc else new_state, None

    if use_remat:
        layer_step = jax.checkpoint(layer_step)
    final_carry, _ = jax.lax.scan(
        layer_step, (state, rng_key) if is_mc else state, params
    )
    final_state = final_carry[0] if is_mc else final_carry

    if is_noisy and not is_mc:
        probs = jnp.real(jnp.diag(cast("JaxF64", final_state)))
    else:
        probs = jnp.abs(cast("JaxF64", final_state)) ** 2
    return probs / (jnp.sum(probs) + 1e-12)


def _step_logic(
    state: tuple[JaxF64, JaxKey | None],
    step_input_unitaries: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    leak_rate: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
) -> tuple[tuple[JaxF64, JaxKey | None], JaxF64]:
    m_prev, rng_key = state  # m_prev: leaky-integrated feedback vector (N,)
    step_key = None
    next_key = None
    if rng_key is not None:
        step_key, next_key = jax.random.split(rng_key)
    probs = _make_circuit_logic(
        step_input_unitaries,
        m_prev,
        reservoir_params,
        n_qubits,
        feedback_scale,
        noise_type,
        noise_prob,
        use_remat,
        use_reuploading,
        step_key,
    )
    output = jnp.dot(measurement_matrix, probs)
    # Leaky integrator: m_n = (1 - λ) * m_{n-1} + λ * ⟨Z_n⟩
    z_expectations = output[:n_qubits]
    new_m = (1.0 - leak_rate) * m_prev + leak_rate * z_expectations
    return (new_m, next_key), output


# --- Public API Wrappers ---


@partial(
    jax.jit, static_argnames=["n_qubits", "noise_type", "use_remat", "use_reuploading"]
)
def _step_jit(
    state: tuple[JaxF64, JaxKey | None],
    input_slice: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    leak_rate: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
) -> tuple[tuple[JaxF64, JaxKey | None], JaxF64]:
    # Per-qubit input diversification
    input_unitaries = _get_input_unitaries(input_slice, n_qubits)
    return _step_logic(
        state,
        input_unitaries,
        reservoir_params,
        measurement_matrix,
        n_qubits,
        feedback_scale,
        leak_rate,
        noise_type,
        noise_prob,
        use_remat,
        use_reuploading,
    )


@partial(
    jax.jit, static_argnames=["n_qubits", "noise_type", "use_remat", "use_reuploading"]
)
def _chunk_scan_jit(
    state_carry: tuple[JaxF64, JaxKey | None],
    chunk_input_unitaries: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    leak_rate: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
) -> tuple[tuple[JaxF64, JaxKey | None], JaxF64]:
    step_func = partial(
        _step_logic,
        reservoir_params=reservoir_params,
        measurement_matrix=measurement_matrix,
        n_qubits=n_qubits,
        feedback_scale=feedback_scale,
        leak_rate=leak_rate,
        noise_type=noise_type,
        noise_prob=noise_prob,
        use_remat=use_remat,
        use_reuploading=use_reuploading,
    )

    in_axs = (0, 0) if chunk_input_unitaries.ndim == 5 else (0, None)
    return jax.lax.scan(
        jax.vmap(step_func, in_axes=in_axs), state_carry, chunk_input_unitaries
    )


def _forward_jit(
    state_init: tuple[JaxF64, JaxKey | None],
    inputs_time_major: JaxF64,
    reservoir_params: JaxF64,
    measurement_matrix: JaxF64,
    n_qubits: int,
    feedback_scale: float,
    leak_rate: float,
    noise_type: str,
    noise_prob: float,
    use_remat: bool,
    use_reuploading: bool,
) -> tuple[tuple[JaxF64, JaxKey | None], JaxF64]:
    T, B, D = inputs_time_major.shape

    # 1. Pre-compute all unitaries for the entire sequence (Time, Batch, N, 4, 4)
    get_step_unitaries = partial(_get_input_unitaries, n_qubits=n_qubits)
    all_input_unitaries = jax.vmap(jax.vmap(get_step_unitaries))(inputs_time_major)

    # 2. Execute full sequence in one shot (No chunking boilerplate)
    final_carry, stacked_outputs = _chunk_scan_jit(
        state_init,
        all_input_unitaries,
        reservoir_params,
        measurement_matrix,
        n_qubits,
        feedback_scale,
        leak_rate,
        noise_type,
        noise_prob,
        use_remat,
        use_reuploading,
    )

    return final_carry, stacked_outputs.reshape(T, B, -1)
