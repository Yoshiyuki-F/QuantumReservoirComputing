import pytest
import subprocess
import sys

# Combinations to test
MODELS: tuple[str, ...] = (
    "passthrough",
    "classical_reservoir",
    "quantum_reservoir", 
    "fnn",
    "fnn_distillation",
)

DATASETS: tuple[str, ...] = (
    "lorenz",
    "mnist",
    "mackey_glass",
)

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_pipeline_execution(model: str, dataset: str) -> None:
    """
    Test that the pipeline accepts the model/dataset combination and runs to completion.
    Equivalent to: uv run python -m reservoir.cli.main --model <model> --dataset <dataset>
    """
    print(f"\n[Test] Running Pipeline: Model={model}, Dataset={dataset}")
    
    # Construct command
    # Using 'uv run' to ensure environment consistency
    command = [
        "uv", "run", "python", "-m", "reservoir.cli.main",
        "--model", model,
        "--dataset", dataset
    ]
    
    try:
        # Run command, capture output, check return code (raises on non-zero)
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[Success] {model} on {dataset}")
        
    except subprocess.CalledProcessError as e:
        print(f"[Failure] {model} on {dataset} exited with code {e.returncode}")
        print("--- Stdout ---")
        print(e.stdout)
        print("--- Stderr ---")
        print(e.stderr)
        pytest.fail(f"Pipeline execution failed for {model} / {dataset}. Error: {e.stderr}")

if __name__ == "__main__":
    # Allow running directly: python tests/integration/test_matrix.py
    # Simple manual runner if pytest is not available
    failed: list[tuple[str, str]] = []
    for m in MODELS:
        for d in DATASETS:
            print(f"Running {m} on {d}...")
            cmd = ["uv", "run", "python", "-m", "reservoir.cli.main", "--model", m, "--dataset", d]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print(f"FAILED: {m} on {d}")
                print(res.stderr)
                failed.append((m, d))
            else:
                print("PASSED")
    
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)
