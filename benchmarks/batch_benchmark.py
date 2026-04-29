"""
Benchmark: Batch Size Throughput
Evaluates throughput (samples/sec) for different batch sizes.
"""
import time
import jax
import jax.numpy as jnp
import pandas as pd
import sys
import os

# Clean path adjustment to include src
sys.path.append(os.path.join(os.getcwd(), "src"))

from reservoir.models.reservoir.quantum.model import QuantumReservoir
from reservoir.layers.aggregation import AggregationMode

def run_benchmark():
    n_qubits = 12 # Moderate size for speed
    time_steps = 50
    n_features = 1 # Just scalar input per time step
    
    # Initialize Model once
    model = QuantumReservoir(
        n_qubits=n_qubits,
        n_layers=3,
        seed=42,
        aggregation_mode=AggregationMode.MEAN,
        feedback_scale=0.1,
        leak_rate=1.0,
        measurement_basis="Z+ZZ",
        noise_type="clean",
        noise_prob=0.0,
        readout_error=0.0,
        n_trajectories=0,
        use_reuploading=False,
    )
    
    # Pre-compile
    dummy_input = jnp.zeros((1, time_steps, n_features))
    dummy_state = model.initialize_state(1)
    print("Compiling...")
    _ = model.forward(dummy_state, dummy_input)
    print("Compilation done.")
    
    batch_sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    results = []
    
    print(f"{'Batch':<8} | {'Time (s)':<10} | {'Throughput (samples/s)':<25}")
    print("-" * 50)
    
    for bs in batch_sizes:
        try:
            inputs = jax.random.normal(jax.random.key(0), (bs, time_steps, n_features))
            state = model.initialize_state(bs)
            
            # Warmup
            _ = model.forward(state, inputs)
            jax.block_until_ready(inputs) # Ensure inputs are ready
            
            # Run
            start = time.time()
            n_iters = 10 # Repeat to average out noise
            for _ in range(n_iters):
                out_state, out_seq = model.forward(state, inputs)
                jax.block_until_ready(out_seq)
                
            elapsed = time.time() - start
            avg_time = elapsed / n_iters
            throughput = bs / avg_time
            
            results.append({
                "BatchDefault": bs, "Time_s": avg_time, "Throughput_Hz": throughput
            })
            print(f"{bs:<8} | {avg_time:<10.4f} | {throughput:<25.2f}")
            
        except (RuntimeError, MemoryError, ValueError) as e:
            print(f"{bs:<8} | {'ERROR':<10} | {str(e)}")
            break # Stop if OOM or error

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_benchmark()
    df.to_csv("benchmarks/batch_results.csv", index=False)
