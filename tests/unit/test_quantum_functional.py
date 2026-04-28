import jax

# Enable 64-bit precision for complex128 before importing jax.numpy
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from reservoir.models.reservoir.quantum.functional import _make_circuit_logic, _get_paper_R_unitary  # noqa: E402
import tensorcircuit as tc  # noqa: E402

# Set dtype and backend globally for tests
tc.set_dtype("complex128")
tc.set_backend("jax")

def get_valid_unitaries(shape):
    # To avoid tc errors, let's just make identity matrices
    L, n_qubits, _, _ = shape
    identity = jnp.eye(2, dtype=jnp.complex128)
    return jnp.tile(identity, (L, n_qubits, 1, 1))

def test_make_circuit_logic_runs():
    tc.set_backend("jax")
    n_qubits = 4
    input_val = jnp.array([0.5, -0.2]) # 2D input
    feedback_val = jnp.array([0.1, 0.2, 0.3, 0.4])
    params = get_valid_unitaries((2, n_qubits, 2, 2)) # 2 layers
    
    input_slice = input_val[jnp.arange(n_qubits) % input_val.shape[0]]
    input_unitaries = jax.vmap(_get_paper_R_unitary)(input_slice)

    probs = _make_circuit_logic(
        input_unitaries=input_unitaries,
        feedback_val=feedback_val,
        params=params,
        n_qubits=n_qubits,
        feedback_scale=1.0,
        noise_type="clean",
        noise_prob=0.0,
        use_remat=False,
        use_reuploading=True,
        rng_key=None
    )
    
    assert probs.shape == (2**n_qubits,)
    print(f"Probabilities sum: {jnp.sum(probs)}")
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
    print("Test passed successfully!")

def test_make_circuit_logic_clean_no_reup():
    tc.set_backend("jax")
    n_qubits = 2
    input_val = jnp.array([0.5])
    feedback_val = jnp.array([0.1, 0.2])
    params = get_valid_unitaries((1, n_qubits, 2, 2))
    
    input_slice = input_val[jnp.arange(n_qubits) % input_val.shape[0]]
    input_unitaries = jax.vmap(_get_paper_R_unitary)(input_slice)

    probs = _make_circuit_logic(
        input_unitaries=input_unitaries,
        feedback_val=feedback_val,
        params=params,
        n_qubits=n_qubits,
        feedback_scale=1.0,
        noise_type="clean",
        noise_prob=0.0,
        use_remat=False,
        use_reuploading=False,
        rng_key=None
    )
    
    s = jnp.sum(probs)
    print(f"[Clean No-Reup] Probabilities sum: {s}")
    assert jnp.allclose(s, 1.0, atol=1e-5)

if __name__ == "__main__":
    test_make_circuit_logic_clean_no_reup()
    test_make_circuit_logic_runs()
