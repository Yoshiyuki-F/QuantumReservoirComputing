import os
import numpy as np
import pandas as pd
import io
from contextlib import redirect_stdout
from reservoir.models.presets import PipelineConfig
from reservoir.models.config import BoundedAffineScalerConfig, QuantumReservoirConfig, PolyRidgeReadoutConfig
from reservoir.models.identifiers import Model
from reservoir.layers.aggregation import AggregationMode
from reservoir.data.identifiers import Dataset
from reservoir.training import get_training_preset
from reservoir.pipelines.run import run_pipeline

scale, rs, f, lr = 0.038015263451324666, 0.19041335672406576, 2.6764652369291992, 0.1732129929337844 # q10 mg

def main():
    lambda_candidates = np.logspace(-12, 3, 30).tolist()
    seeds = [41, 42, 43]
    dataset = Dataset("mackey_glass") 
    
    csv_path = "scripts/experiments/qrc_lambda_seed_results.csv"
    
    for seed in seeds:
        for lam in lambda_candidates:
            print("\n" + "="*50)
            print(f"Running seed={seed}, lambda={lam:.2e}")
            print("="*50 + "\n")
            
            preset = PipelineConfig(
                name="quantum_reservoir",
                model_type=Model.QUANTUM_RESERVOIR,
                description="Quantum Gate-Based Reservoir Computing (Time Series)",
                preprocess=BoundedAffineScalerConfig(
                    scale=scale,
                    relative_shift=rs,
                    bound=np.pi,
                ),
                projection=None,
                model=QuantumReservoirConfig(
                    n_qubits=10,
                    n_layers=1,
                    seed=seed,
                    aggregation=AggregationMode.SEQUENCE,
                    feedback_scale=f,
                    leak_rate=lr,
                    measurement_basis="Z+ZZ",
                    noise_type="clean",
                    noise_prob=0.0,
                    readout_error=0.0,
                    n_trajectories=0,
                    use_reuploading=True,
                ),
                readout=PolyRidgeReadoutConfig(
                    use_intercept=False,
                    lambda_candidates=(lam,), 
                    degree=2,
                    mode="square_only",
                    norm_threshold=4000.0,
                ),
            )
            
            training_config = get_training_preset("quantum")
            
            # Capture output to check for divergence warnings
            f_out = io.StringIO()
            error_occurred = None
            res = {}
            with redirect_stdout(f_out):
                try:
                    res = run_pipeline(preset, dataset, training_config)
                except Exception as e:
                    error_occurred = str(e)
            
            # Print the output that we captured
            output = f_out.getvalue()
            print(output)
            
            diverged = "Closed-loop prediction diverged!" in output
            
            vpt_lt = None
            if not error_occurred:
                # Correctly extract vpt_lt from the reporter's output structure
                vpt_lt = res.get("test", {}).get("vpt_lt")

            row = {
                "seed": seed,
                "lambda": lam,
                "vpt_lt": vpt_lt,
                "diverged": diverged,
                "error": error_occurred
            }
            print(f"Result -> VPT_LT: {vpt_lt}, Diverged: {diverged}, Error: {error_occurred is not None}")
            
            # Save to CSV incrementally so we don't lose data
            df = pd.DataFrame([row])
            header = not os.path.exists(csv_path)
            df.to_csv(csv_path, mode='a', header=header, index=False)

if __name__ == "__main__":
    main()
