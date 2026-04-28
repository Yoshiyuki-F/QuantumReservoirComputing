"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/factory.py
STEP 4, 5, 6 read README.md
Global entry point for model creation. Delegates to specialized factories.
"""
from __future__ import annotations


from reservoir.models.nn.fnn import FNNModel
from reservoir.models.identifiers import Model
from reservoir.models.config import ClassicalReservoirConfig, DistillationConfig, FNNConfig, QuantumReservoirConfig
from reservoir.models.distillation.factory import DistillationFactory
from reservoir.models.reservoir.factory import ReservoirFactory
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from reservoir.training.presets import TrainingConfig
    from reservoir.core.types import TopologyMeta
    from reservoir.models.presets import PipelineConfig
    from reservoir.models.generative import ClosedLoopModel


class ModelFactory:
    """Router that delegates model construction to specialized factories."""

    @staticmethod
    def create_model(
        config: PipelineConfig,
        training: TrainingConfig | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
        input_shape: tuple[int, ...] | None = None,
        classification: bool = False,
    ) -> ClosedLoopModel:

        if input_dim is None or input_dim <= 0:
            raise ValueError("ModelFactory.create_model requires a positive input_dim.")
        if output_dim is None or output_dim <= 0:
            raise ValueError("ModelFactory.create_model requires a positive output_dim.")

        training_cfg = training
        pipeline_enum = config.model_type

        if pipeline_enum in {Model.CLASSICAL_RESERVOIR, Model.QUANTUM_RESERVOIR}:
            if not isinstance(config.model, (ClassicalReservoirConfig, QuantumReservoirConfig)):
                raise TypeError(f"Reservoir pipelines require ClassicalReservoirConfig or QuantumReservoirConfig, got {type(config.model)}.")

            return ReservoirFactory.create_model(
                pipeline_config=config,
                projected_input_dim=input_dim,
                output_dim=output_dim,
                input_shape=input_shape,
            )

        if pipeline_enum in {Model.FNN_DISTILLATION_CLASSICAL, Model.FNN_DISTILLATION_QUANTUM}:
            if not isinstance(config.model, DistillationConfig):
                raise TypeError(f"fnn_distillation_classical pipeline requires DistillationConfig, got {type(config.model)}.")
            return DistillationFactory.create_model(
                distillation_config=config.model,
                training=training_cfg,
                input_dim=input_dim,
                output_dim=output_dim,
                input_shape=input_shape,
            )

        if pipeline_enum == Model.FNN:
            if not isinstance(config.model, FNNConfig):
                raise TypeError(f"FNN pipeline requires FNNConfig, got {type(config.model)}.")

            # Check if windowed mode
            window_size = config.model.window_size
            
            # Calculate correct dimensions
            batch_size = None
            timesteps = None
            if input_shape:
                batch_size = int(input_shape[0])
                if len(input_shape) > 1:
                    timesteps = int(input_shape[1])
            
            windowed_samples = None
            if window_size is not None:
                # Windowed FNN: flattened_dim = window_size * input_dim
                flattened_dim = window_size * int(input_dim)
                # Number of samples after windowing
                if timesteps:
                    windowed_samples = timesteps - window_size + 1
            else:
                # Standard FNN: flatten all timesteps
                flattened_dim = timesteps * int(input_dim) if timesteps else int(input_dim)

            model = FNNModel(
                model_config=config.model,
                training_config=training_cfg,
                input_dim=int(flattened_dim),  # Effective input dimension (flattened/windowed)
                output_dim=output_dim,
                classification=classification,
            )
            
            # Attach topology metadata for FNN
            hidden_layers = tuple(int(h) for h in (config.model.hidden_layers or ()) if int(h) > 0)
            
            # Build adapter string and shape
            if window_size is not None:
                adapter_shape = (windowed_samples, flattened_dim) if windowed_samples else (flattened_dim,)
                structure = f"TimeDelayEmbedding(K={window_size}) -> FNN -> Output"
            else:
                adapter_shape = (batch_size, flattened_dim) if batch_size else (flattened_dim,)
                structure = "Flatten -> FNN -> Output"
            
            feature_rows = windowed_samples if windowed_samples is not None else batch_size
            feature_shape = (feature_rows, output_dim) if feature_rows is not None else (output_dim,)

            topo_meta: TopologyMeta = {
                "type": pipeline_enum.value.upper(),
                "shapes": {
                    "input": input_shape,
                    "projected": input_shape,
                    "adapter": adapter_shape,
                    "internal": tuple(hidden_layers) if hidden_layers else None,
                    "feature": feature_shape,
                    "output": feature_shape,
                },
                "details": {
                    "window_size": window_size,
                    "student_layers": hidden_layers or None,
                    "structure": structure,
                    "agg_mode": "None",
                    "readout": "None",
                },
            }
            model.topology_meta = topo_meta
            return model

        if pipeline_enum == Model.PASSTHROUGH:
            from reservoir.models.config import PassthroughConfig
            if not isinstance(config.model, PassthroughConfig):
                raise TypeError(f"PASSTHROUGH pipeline requires PassthroughConfig, got {type(config.model)}.")
            from reservoir.models.passthrough.factory import PassthroughFactory
            return PassthroughFactory.create_model(
                pipeline_config=config,
                projected_input_dim=input_dim,
                output_dim=output_dim,
                input_shape=input_shape,
            )

        raise ValueError(f"Unsupported model_type: {pipeline_enum}")
