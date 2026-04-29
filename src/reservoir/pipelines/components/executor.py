# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/components/executor.py

import jax.numpy as jnp
from typing import cast, TYPE_CHECKING
from reservoir.core.types import JaxF64, ModelState, NpF64, to_jax_f64, to_np_f64

from reservoir.pipelines.evaluation import Evaluator
from reservoir.pipelines.strategies import ReadoutStrategyFactory
from reservoir.utils.reporting import print_feature_stats
from reservoir.models.reservoir.base import Reservoir
from reservoir.core.types import ResultDict

if TYPE_CHECKING:
    from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
    from reservoir.models.generative import ClosedLoopGenerativeModel, ClosedLoopModel
    from reservoir.pipelines.components.data_coordinator import DataCoordinator
    from reservoir.models.presets import PipelineConfig


class PipelineExecutor:
    """
    Orchestrates the pipeline execution.
    Acts as a Site Supervisor:
    1. Asks DataCoordinator for materials (Data).
    2. Directs Worker (Model) to process materials.
    3. Asks StrategyFactory for specialized tooling (Readout Strategy).
    4. Executes the finish work.
    """

    def __init__(self, stack: ModelStack, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata, coordinator: DataCoordinator):
        self.stack = stack
        self.frontend_ctx = frontend_ctx
        self.dataset_meta = dataset_meta
        self.evaluator = Evaluator()
        # Dependency Injection (OCP/DIP)
        self.coordinator = coordinator

    def run(self, config: PipelineConfig) -> ResultDict:
        """Run the execution phase."""
        
        # Step 5: Model Dynamics (Training/Warmup)
        # Coordinator provides raw training data
        train_X = self.coordinator.get_train_inputs()
        projection = self.frontend_ctx.projection_layer
        
        train_y = self.frontend_ctx.processed_split.train_y
            
        train_logs = self.stack.model.train(
            to_jax_f64(train_X), 
            to_jax_f64(train_y) if train_y is not None else None,
            projection_layer=projection,
        )


        # Delegate extraction (Model does work, Coordinator provides input)
        print("\n[executor.py] === Step 5.5: Extract Features (Output) ===")
        train_Z, val_Z, test_Z, val_final_info = self._extract_all_features(self.stack.model)
        if train_Z is None:
            raise ValueError("Training feature extraction returned None.")

        # Step 6.5: Target Alignment (Delegate to Coordinator)
        print("\n=== Step 6.5: Target Alignment (Auto-Align) ===")
        train_y = self.coordinator.align_targets(train_Z, "train")
        val_y = self.coordinator.align_targets(val_Z, "val")
        
        # If test_Z was skipped, test_y still needs to be aligned to the test set length
        if test_Z is not None:
            test_y = self.coordinator.align_targets(test_Z, "test")
        else:
            # Fallback: test_y should match the length of the processed test_X
            test_y = self.frontend_ctx.processed_split.test_y
            print(f"    [Executor] Using raw test_y (Length: {len(test_y) if test_y is not None else 0})")

        # Step 7: Fit Readout (Strategy Pattern)
        readout_name = self.stack.readout.__class__.__name__ if self.stack.readout else "None"
        print(f"\n=== Step 7: Readout ({readout_name}) with val data ===")
        
        strategy = ReadoutStrategyFactory.create_strategy(
            self.stack.readout,
            self.dataset_meta, 
            self.evaluator, 
            self.stack.metric
        )

        fit_result = strategy.fit_and_evaluate(
            self.stack.model,
            self.stack.readout,
            train_Z, val_Z, test_Z,
            train_y, val_y, test_y,
            self.frontend_ctx,
            self.dataset_meta,
            config,
            topo_meta=self.stack.topo_meta,
            val_final_state=val_final_info
        )
        
        # Step 8.5: Capture Quantum Trace (for Visualization)
        quantum_trace = None
        # Check if it's a QuantumReservoir (duck typing or class check)
        if hasattr(self.stack.model, "n_qubits") and hasattr(self.stack.model, "measurement_basis"):
             print("\n=== Step 8.5: Capturing Quantum Dynamics Trace ===")
             try:
                 if fit_result.get("closed_loop_history") is not None:
                     # Use history from closed-loop generation (no re-computation needed)
                     quantum_trace = to_np_f64(cast("JaxF64", fit_result["closed_loop_history"]))
                     print(f"    [Executor] Using captured closed-loop history: {quantum_trace.shape}")
                 else:
                     test_data = self.frontend_ctx.processed_split.test_X
                     if test_data is None:
                         test_data = self.frontend_ctx.processed_split.train_X
                         
                     sample_input = None
                     if test_data is not None and len(test_data) > 0:
                          # Heuristic: If 2D (Batch, Feat), treat as sequence of length min(100, N)
                          if test_data.ndim == 2:
                               # Use full sequence for visualization as requested
                               sample_input = test_data[None, :, :]
                          # If 3D (Batch, Time, Feat), take first sample -> (1, T, F)
                          elif test_data.ndim == 3:
                               sample_input = test_data[0:1]
                     
                     if sample_input is not None:
                         # Force return_sequences=True to get time evolution
                         # Use batch_size=1
                         # Convert to JAX array to satisfy strictly typed model
                         sample_input_jax = to_jax_f64(sample_input)
                         trace = self.stack.model(sample_input_jax, return_sequences=True)
                         quantum_trace = to_np_f64(trace)
                         print(f"[executor.py] Captured trace shape: {trace.shape}")
             except (ValueError, RuntimeError, TypeError) as e:
                 print(f"[executor.py] Failed to capture quantum trace: {e}")

        result: ResultDict = {
            "fit_result": fit_result,
            "train_logs": train_logs,
            "quantum_trace": quantum_trace,
        }
        return result

    def _extract_all_features(self, model: ClosedLoopModel) \
            -> tuple[NpF64 | None, NpF64 | None, NpF64 | None, tuple[ModelState | None, JaxF64 | None] | None]:
        """
        Orchestrate feature extraction using DataLoader.
        """

        window_size = getattr(model, 'input_window_size', 0)
        batch_size = self.dataset_meta.training.batch_size
        projection = self.frontend_ctx.projection_layer  # May be None
        
        is_stateful = isinstance(model, Reservoir)
        
        def process_split(
            split_name: str,
            initial_state: ModelState | None = None,
            return_state: bool = False,
        ) -> tuple[NpF64 | None, ModelState | None, JaxF64 | None]:
            loader = self.coordinator.get_eval_dataloader(split_name, batch_size, window_size, projection)
            if loader is None or loader.num_samples == 0:
                return None, None, None

            if return_state and hasattr(loader, "X") and loader.X is not None and loader.X.ndim == 2:
                # Stateful recurrent generation requires entire sequence at once
                for b_x, _ in loader:
                    inputs_jax = b_x[None, :, :]
                    model_for_state = cast("ClosedLoopGenerativeModel[ModelState]", model)
                    state = initial_state if initial_state is not None else model_for_state.initialize_state(1)
                    final_state, outputs_jax = model_for_state.forward(state, inputs_jax)
                    outputs_np = to_np_f64(outputs_jax[0])
                    last_output = outputs_jax[:, -1, :] 
                    print_feature_stats(outputs_np, "executor.py", f"5.5:Z:{split_name}")
                    return outputs_np, final_state, last_output

            # Standard Path using generator
            import numpy as np
            from tqdm.auto import tqdm
            all_outputs: list[NpF64] = []
            
            pbar = tqdm(total=loader.num_samples, desc=f"Extracting {split_name}", unit="samples")
            for b_x, _ in loader:
                b_out = model(b_x)
                all_outputs.append(to_np_f64(b_out))
                pbar.update(b_x.shape[0])
            pbar.close()
            
            outputs_np = np.concatenate(all_outputs, axis=0) if all_outputs else None
            if outputs_np is not None:
                print_feature_stats(outputs_np, "executor.py", f"5.5:Z:{split_name}")
            return outputs_np, None, None

        current_state: ModelState | None = None
        if is_stateful:
            warmup_features, current_state, _ = process_split("warmup", initial_state=current_state, return_state=is_stateful)
            if warmup_features is not None:
                warmup_std = float(jnp.std(warmup_features))
                if warmup_std < 0.1 and not self.dataset_meta.classification:
                    raise ValueError(f"Feature collapse detected! warmup_X std ({warmup_std:.4f}) < 0.1. "
                                     "This usually indicates the Reservoir state is saturated or not responding to input.")

        # 1. Train
        train_Z, current_state, _ = process_split("train", initial_state=current_state, return_state=is_stateful)

        # 2. Validation
        val_Z, current_state, val_last_output = process_split("val", initial_state=current_state, return_state=is_stateful)

        if val_Z is not None:
            val_std = float(jnp.std(val_Z))
            if val_std < 0.1:
                 print(f"    [Warning] val_Z std ({val_std:.4f}) is very low.")

        val_final_state = current_state

        if is_stateful:
             state_stat = "Captured" if val_final_state is not None else "None"
             print(f"    [Executor] Reservoir State Chaining: {state_stat}")

        # 3. Test
        test_Z = None
        if self.dataset_meta.classification:
            test_Z, _, _ = process_split("test", initial_state=None, return_state=False)

            if test_Z is not None:
                test_std = float(jnp.std(test_Z))
                if test_std < 0.1:
                    print(f"    [Warning] test_Z std ({test_std:.4f}) is very low.")
        else:
            print("    [Executor] Skipping Test feature extraction for Regression task (Closed-loop will be used).")
            
        return train_Z, val_Z, test_Z, (val_final_state, val_last_output)
