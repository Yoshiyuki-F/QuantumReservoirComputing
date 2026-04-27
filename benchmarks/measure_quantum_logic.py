import time
import jax
import jax.numpy as jnp
from reservoir.models.reservoir.quantum.functional import _make_circuit_logic, _get_fused_rotation_matrix, _get_paper_R_unitary

# Ensure JAX is in float64 mode
jax.config.update("jax_enable_x64", True)

def benchmark_quantum_circuit():
    n_qubits = 14
    n_layers = 10
    
    input_val = jnp.array([0.5, -0.2, 0.1], dtype=jnp.float64)
    feedback_val = jnp.zeros(n_qubits, dtype=jnp.float64)
    
    print("Pre-computing static unitaries (complex128)...")
    raw_params = jax.random.uniform(jax.random.key(0), (n_layers, n_qubits, 3), dtype=jnp.float64)
    v_get_matrix = jax.vmap(jax.vmap(_get_fused_rotation_matrix))
    params = v_get_matrix(raw_params)
    
    input_dim = input_val.shape[0]
    step_input_unitaries = jnp.stack([_get_paper_R_unitary(input_val[i % input_dim]) for i in range(n_qubits)])

    print("--- Benchmarking Quantum Logic (Batched Indexed + complex128) ---")
    
    @jax.jit
    def bench_loop(iu, fv, p):
        def body(carry, _):
            # _make_circuit_logic(iu, fv, p, n_qubits, 1.0, use_reuploading=True)
            res = _make_circuit_logic(
                input_unitaries=iu,
                feedback_val=fv,
                params=p,
                n_qubits=n_qubits,
                feedback_scale=1.0,
                noise_type="clean",
                noise_prob=0.0,
                use_remat=False,
                use_reuploading=True,
                rng_key=None
            )
            return carry, res
        return jax.lax.scan(body, None, jnp.arange(100))

    # Warm-up
    print("Compiling Benchmark Loop...")
    start_c = time.time()
    _, res = bench_loop(step_input_unitaries, feedback_val, params)
    res.block_until_ready()
    print(f"Compilation took: {time.time() - start_c:.4f}s")
    
    n_runs = 100
    start = time.time()
    _, res = bench_loop(step_input_unitaries, feedback_val, params)
    res.block_until_ready()
    total_time = time.time() - start
    
    avg_time = total_time / n_runs
    print(f"Total time for {n_runs} runs: {total_time:.4f}s")
    print(f"Average time per step: {avg_time*1000:.4f}ms")
    print(f"Estimated time for 10,000 steps: {avg_time * 10000:.2f}s")

if __name__ == "__main__":
    benchmark_quantum_circuit()
