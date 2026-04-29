"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/distillation/factory.py
STEP 4 and 5 (6 is skipped)
Factory for building distillation teacher-student pipelines."""
from __future__ import annotations



from reservoir.models.identifiers import Model
from reservoir.models.nn.fnn import FNNModel
from reservoir.models.distillation.model import DistillationModel
from reservoir.models.reservoir.classical import ClassicalReservoir
from reservoir.models.config import QuantumReservoirConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import TopologyMeta
    from reservoir.training.presets import TrainingConfig
    from reservoir.models.config import DistillationConfig


class DistillationFactory:
    """Builds teacher (reservoir pipeline) and student (FNN) for distillation."""

    @staticmethod
    def create_model(
        distillation_config: DistillationConfig,
        input_dim: int,
        output_dim: int,
        input_shape: tuple[int, ...] | None,
        training: TrainingConfig | None = None,
    ) -> DistillationModel:
        teacher_cfg = distillation_config.teacher
        teacher_cfg.validate(context="distillation.teacher")

        projected_input_dim = int(input_dim)
        if projected_input_dim <= 0:
            raise ValueError(f"input_dim must be positive for distillation, got {input_dim}")

        if input_shape is None:
            raise ValueError("input_shape must be provided for distillation (time, features).")
        
        # Check input_shape dimensionality
        # Expect (Batch, Time, Features) -> 3D
        if len(input_shape) < 3:
             # Fallback or error? Assuming 3D for distillation of sequences
             time_steps = 1
             batch_size = int(input_shape[0])
        else:
             batch_size = int(input_shape[0])
             time_steps = int(input_shape[1])

        #1. create teacher (Classical or Quantum)
        if isinstance(teacher_cfg, QuantumReservoirConfig):
            from reservoir.models.reservoir.quantum.model import QuantumReservoir

            n_qubits = projected_input_dim  # Projection output feeds directly as qubits
            teacher_node = QuantumReservoir(
                n_qubits=n_qubits,
                n_layers=teacher_cfg.n_layers,
                seed=teacher_cfg.seed,
                aggregation_mode=teacher_cfg.aggregation,
                feedback_scale=teacher_cfg.feedback_scale,
                leak_rate=teacher_cfg.leak_rate,
                measurement_basis=teacher_cfg.measurement_basis,
                noise_type=teacher_cfg.noise_type,
                noise_prob=teacher_cfg.noise_prob,
                readout_error=teacher_cfg.readout_error,
                n_trajectories=teacher_cfg.n_trajectories,
                use_reuploading=teacher_cfg.use_reuploading,
            )
        else:
            teacher_node = ClassicalReservoir(
                n_units=projected_input_dim,
                spectral_radius=teacher_cfg.spectral_radius,
                leak_rate=teacher_cfg.leak_rate,
                rc_connectivity=teacher_cfg.rc_connectivity,
                seed=teacher_cfg.seed,
                aggregation_mode=teacher_cfg.aggregation,
            )
        teacher_feature_dim = teacher_node.get_feature_dim(time_steps=time_steps)

        #2. configure student FNN input dimension based on window or flatten
        # Use RAW input feature dimension (e.g., 28 for MNIST) instead of projected dim (100)
        # Distillation typically trains the student on RAW inputs to mimic the whole pipeline.
        student_raw_feat_dim = int(input_shape[-1]) if input_shape else projected_input_dim
        
        window_size = distillation_config.student.window_size
        if window_size is not None:
            student_input_dim = student_raw_feat_dim * window_size
        else:
            student_input_dim = time_steps * student_raw_feat_dim
        
        h_layers = distillation_config.student.hidden_layers
        hidden_layers = list(h_layers or [])

        #3. create student (FNNModel handles adapter internally based on window_size)
        # student_input_dim is the EFFECTIVE dimension (e.g. 784 for MNIST 28*28)
        student_model = FNNModel(
            model_config=distillation_config.student,
            training_config=training,
            input_dim=int(student_input_dim),  # Flattened/windowed dimension (e.g., 784)
            output_dim=int(teacher_feature_dim),
            classification=False,
        )

        model = DistillationModel(
            teacher=teacher_node,
            student=student_model,
            training_config=training,
        )

        is_quantum = isinstance(teacher_cfg, QuantumReservoirConfig)
        pipeline_type = Model.FNN_DISTILLATION_QUANTUM if is_quantum else Model.FNN_DISTILLATION_CLASSICAL
        topo_meta: TopologyMeta = {
            "type": pipeline_type.value.upper(),
            "shapes": {
                "input": input_shape,
                "preprocessed": None,  # preprocessing happens upstream
                "projected": (batch_size, time_steps, projected_input_dim),  # sequence into flatten
                "adapter": (batch_size, student_input_dim),  # flattened time-major input to FNN
                "internal": tuple(hidden_layers) if hidden_layers else None,  # hidden layer widths
                "feature": (batch_size, int(teacher_feature_dim)),  # student output equals teacher feature dim
                "output": (batch_size, output_dim),  # readout target size
            },
            "details": {
                "preprocess": "TDE",
                "agg_mode": "None",
                "student_layers": tuple(hidden_layers) if hidden_layers else None,
                "student_structure": f"TDE(w={window_size}) -> FNN",
            },
        }
        model.topology_meta = topo_meta
        return model
