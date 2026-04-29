# Quantum Reservoir Computing (QRC) Benchmark & Optimizations

This document records the comprehensive performance benchmarks, architectural optimizations, and technical findings of the Quantum Reservoir simulator (JAX + TensorCircuit backend).

## 1. System Configuration
- **Backend:** JAX (XLA) + TensorCircuit
- **Precision:** `complex128` (Strict requirement for numerical stability and high-fidelity research)
- **Hardware Target:** Optimized for high-throughput BLAS/cuBLAS via JAX-XLA.

## 2. Optimization History & Findings

### Phase 1: Support for Multi-dimensional Input
- **Implementation:** **Repeated Encoding**.
- **Change:** Multi-variate input $\mathbf{u}_{n+1}$ is cyclically mapped across all $N$ qubits instead of using only the first dimension.
- **Effect:** Full information retention for datasets like Lorenz (3D).

### Phase 2: Fundamental Speedup (Gate Fusion & Pre-computation)
- **Optimization:** Fused single-qubit rotations (RX-RY-RZ) into 2x2 unitaries and complex Paper R gates into 4x4 unitaries.
- **Static Unitary Pre-computation:** Reservoir rotation matrices are computed once during initialization.
- **Sequence Pre-computation:** Input unitaries for the entire time series (sequence) are computed in one batch before the simulation loop starts.
- **Effect:** Moved thousands of matrix generation calls from the hot loop to initialization/setup phases.

### Phase 3: Entanglement Parallelization
- **Optimization:** **Brickwork Parallelization**.
- **Method:** Replaced the sequential CNOT ladder (0-1, 1-2, 2-3...) with parallel blocks (Even pairs then Odd pairs).
- **Effect:** Reduced logical circuit depth from $N$ to $2$, significantly improving XLA kernel utilization and throughput.

### Phase 4: Measurement Engine Optimization (Benchmarking Results)
Multiple strategies were tested to extract $Z$ and $ZZ$ expectations:
- **Matrix-Vector Dot Product ($M \cdot p$):** Winner. Leverages highly optimized BLAS/cuBLAS libraries.
- **Tensor Marginalization (Reshape/Sum):** Competitive but less flexible for arbitrary observables.

### Phase 5: Deep Framework Optimization & Theoretical Limits
Attempted to push beyond the high-level `tc.Circuit` API to reach the physical speed limit of JAX:
1. **Pure JAX Engine (`tensordot` + `moveaxis`):**
   - Result: **~52 ms/step** (Significant Degradation).
   - Finding: Manual axis manipulation in JAX causes massive memory-stride overhead. TensorCircuit's internal engine is highly specialized for these patterns and outperforms generic JAX tensor operations.
2. **Low-level Backend Bypass:**
   - Finding: JIT-compiled `tc.Circuit` objects are nearly as fast as raw backend calls because XLA eliminates the class overhead during compilation.

### Phase 6: Domain-Specific Constraints & Rejected Optimizations
- **Hypothesis:** Bypassing feedback logic when `feedback_scale == 0` and removing the backward CNOT ladder to prevent cross-layer entanglement collapse.
- **Finding:** This architecture strictly enforces `use_reuploading = True` and always uses feedback (`feedback_scale > 0`). Because reuploading unitaries are injected between layers, the backward and forward CNOTs do not cancel each other out.
- **Conclusion:** **Rejected**. Zero-feedback bypass is a dead code path, and modifying the CNOT ladder would alter the intended entanglement topology unnecessarily.

### Phase 7: Deep Unrolling & Unitary Fusion (Feedback-Heavy Scenarios)
- **Optimization 1: Analytical Paper R Gate:** The `get_paper_R_unitary` function originally relied on dynamic `jnp.kron` and `jnp.matmul` operations to construct the $4 \times 4$ Paper R matrix. This was replaced with a hardcoded, analytically derived $4 \times 4$ matrix composed directly from $\cos(\theta/2)$ and $\sin(\theta/2)$.
  - **Effect:** Completely eliminated expensive runtime matrix multiplications and tensor Kronecker products.
- **Optimization 2: Input and Feedback Fusion:** In the encoding step, the input unitary $U_{in}$ on $(i, (i+1)\%N)$ and feedback unitary $U_{fb}$ on $((i+1)\%N, i)$ were previously applied to the state vector sequentially as two separate operations.
  - **Method:** Used matrix index swapping to align the target qubits and pre-multiplied them into a single fused unitary ($U_{fused} = U_{fb\_swapped} \cdot U_{in}$).
  - **Effect:** Halved the number of heavy $4 \times 4$ state-vector applications during the encoding phase.
- **Result:** Improved the batched JIT compilation execution time from ~29.38 ms/step down to ~28.90 ms/step, while also resolving implicit type-casting warnings.

### Phase 8: Strict JAX Purity & PRNG Vectorization (Monte Carlo)
- **Optimization 1: Purity Enforcement:** Removed `_ensure_tensorcircuit_initialized()` and `tc.backend.convert_to_tensor(m)` from within `vmap`/`jit` pure functions (`get_paper_R_unitary`, `get_fused_rotation_matrix`).
  - **Effect:** Eliminated JAX tracing overhead and potential side-effect leakage by returning pure `jnp.ndarray` directly.
- **Optimization 2: Vectorized PRNG Splits:** The Monte Carlo noise application (`is_mc=True`) previously suffered from sequential `jax.random.split` calls inside a python `for` loop.
  - **Method:** Moved the split outside the loop to generate an array of required random numbers at once: `r_array = jax.random.uniform(k1, shape=(len(indices),))`.
  - **Effect:** Massively reduced compilation and execution overhead during noisy Monte Carlo simulations by avoiding sequential state-key tracing.

### Phase 9: Mathematical Redundancy Elimination
- **Optimization 1: Transcendental Reduction:** The calculation of $e^{\pm i\theta/2}$ for the Paper R gate (`get_paper_R_unitary`) and the RZ gate (`get_fused_rotation_matrix`) required high-cost `jnp.exp` operations despite already having computed $\cos(\theta/2)$ and $\sin(\theta/2)$.
  - **Method:** Applied Euler's formula to rewrite these terms purely as $c \pm s$ where $c = \cos(\theta/2), s = -i\sin(\theta/2)$, completely eliminating all $e^x$ function evaluations per unitary per step.
- **Optimization 2: SWAP Invariance (No-op Removal):** The feedback matrix $U_{fb}$ required a $2 \times 2$ quantum swap to align the target qubits `(i+1, i)` to match the input `(i, i+1)`. This involved Python-level index tensor slicing (`swap_idx`).
  - **Finding:** A mathematical proof established that the $4 \times 4$ Paper R matrix is perfectly symmetric under SWAP (`SWAP @ M @ SWAP == M`).
  - **Method:** Removed `swap_idx` and the associated 2D slice (`fb_swapped = fb_unitaries[:, swap_idx, :][:, :, swap_idx]`).
  - **Effect:** Reduced JIT compile complexity and memory reallocation during runtime by eliminating a heavy array manipulation that did not alter the mathematical outcome.

## 3. Final Benchmarks (N=14 Qubits, 10 Layers, complex128)

| Engine Version | Step Time (ms) | Speedup | Status |
| :--- | :---: | :---: | :--- |
| Initial (Sequential TC) | ~9.44 | 1.0x | Obsolete |
| Static Pre-computation | ~6.50 | 1.45x | Obsolete |
| Pure JAX (Manual) | ~52.00 | 0.18x | **Rejected** (Memory overhead) |
| **Final Optimized (Static + Brickwork + TC + Math Reductions)** | **~2.57** | **~3.67x** | **ACTIVE** |

**Final Performance Summary:**
- **Step Throughput:** ~3,890 steps/sec
- **10,000 Step Simulation:** **~25.7 seconds**
- **Architecture:** TensorCircuit-based state evolution with dual-layer Brickwork entanglement.
- **Stability:** **Strict Normalization** enforced to maintain $1.0$ probability sum across long sequences.

## 4. Verification & Testing
To ensure the correctness and performance of the optimized QRC engine, the following commands should be executed after any modifications:

**1. Functional Correctness:**
Validates the probability normalizations and mathematical integrity of the quantum logic:
```bash
uv run pytest tests/unit/test_quantum_functional.py
```

**2. Performance Benchmarking:**
Measures the JIT-compiled execution speed (ms/step) to verify that performance targets are met:
```bash
uv run python benchmarks/measure_quantum_logic.py
```

## 5. Engineering Conclusion
The most efficient path for high-qubit simulation ($N \geq 14$) in JAX is not manual tensor manipulation, but **highly optimized domain-specific frameworks (TensorCircuit)** combined with **static parameter pre-computation** and **parallel circuit structures (Brickwork)**.

---
*Documented by: Yoshi & Gemini CLI*
*Last Updated: February 26, 2026*


Q16 T=8300
246.3215s 
kai4 - 234.0121s
