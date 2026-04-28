from __future__ import annotations

from beartype import beartype
import jax
import jax.numpy as jnp

from reservoir.models.generative import ClosedLoopGenerativeModel
from reservoir.models.reservoir.base import ReservoirModel  # noqa: TC001

from reservoir.core.types import JaxF64, TrainLogs, EvalMetrics, KwargsDict, TopologyMeta
from reservoir.core.types import to_np_f64
from reservoir.core.types import to_jax_f64
from reservoir.models.nn.fnn import FNNModel  # noqa: TC001
from reservoir.training.presets import TrainingConfig  # noqa: TC001
from reservoir.utils.batched_compute import batched_compute
from reservoir.layers.projection import Projection  # noqa: TC001


@beartype
class DistillationModel(ClosedLoopGenerativeModel[JaxF64]):
    """
    Distills reservoir dynamics into a student FNN.
    Teacher: ClassicalReservoir (handles aggregation internally; no gradients).
    Student: FNNModel trained to regress onto teacher targets.
    
    Implements ClosedLoopGenerativeModel to allow autonomous closed-loop generation.
    State representation: Sliding window of input features.
    """

    @property
    def input_window_size(self) -> int:
        return self._window_size

    def __init__(
        self,
        teacher: ReservoirModel,
        student: FNNModel,
        training_config: TrainingConfig | None = None,
    ):
        self.teacher = teacher
        self.student = student
        self.training_config = training_config
        
        self._input_dim: int | None = None
        self._window_size: int = getattr(student, 'window_size', 1) or 1

    def __call__(
        self,
        inputs: JaxF64,
        params: KwargsDict | None = None,
        projection_layer: Projection | None = None,
    ) -> JaxF64:
        return self.predict(inputs, params=params, projection_layer=projection_layer)

    def predict(
        self,
        X: JaxF64,
        params: KwargsDict | None = None,
        projection_layer: Projection | None = None,
    ) -> JaxF64:
        return self.student.predict(X)

    def _compute_teacher_targets(self, inputs: JaxF64, projection_layer: Projection | None = None) -> JaxF64:
        """Legacy single-batch implementation."""
        teacher_inputs = projection_layer(inputs) if projection_layer is not None else inputs
        teacher_outputs = self.teacher(teacher_inputs)
        return jax.lax.stop_gradient(teacher_outputs)

    def train(
        self,
        inputs: JaxF64,
        targets: JaxF64 | None = None,
        log_prefix: str = "4",
        projection_layer: Projection | None = None,
    ) -> TrainLogs:
        """
        Orchestrate the distillation process with clear phase separation in logs.
        """
        print("\n[Distillation] ==========================================")
        print("[Distillation] Phase A: Teacher Target Generation")
        print("[Distillation] ==========================================")

        if self.training_config is None:
            raise ValueError("DistillationModel requires training_config for batching.")
        batch_sz = self.training_config.batch_size
        
        def teacher_fn(x: JaxF64) -> JaxF64:
            x_proj = projection_layer(x) if projection_layer is not None else x
            return jax.lax.stop_gradient(self.teacher(x_proj))

        targets_np = batched_compute(
            teacher_fn,
            to_np_f64(inputs),
            batch_size=batch_sz,
            desc="Teacher (Train set)",
            file="model.py",
            step="3A+5A+6A"
        )
        teacher_targets = to_jax_f64(targets_np)

        print("\n[Distillation] ==========================================")
        print("[Distillation] Phase B: Student Model Training")
        print("[Distillation] ==========================================")
        print(f"[Student] Training {self.student.__class__.__name__} to mimic Teacher...")

        # We pass raw inputs and teacher_targets. 
        # FNNModel will handle the projection inside its JIT loop to avoid OOM.
        student_logs = self.student.train(
            inputs,
            teacher_targets,
            log_prefix="4B",
            projection_layer=projection_layer,
        ) or {}

        distill_mse = float(student_logs.get("final_loss", 0.0))
        student_logs.setdefault("distill_mse", distill_mse)
        student_logs.setdefault("final_loss", distill_mse)
        return student_logs

    def evaluate(
        self,
        X: JaxF64,
        y: JaxF64 | None = None,
        projection_layer: Projection | None = None,
    ) -> EvalMetrics:
        """
        Distillation evaluation: compare student output with teacher output.
        """
        import jax
        if self.training_config is None:
            raise ValueError("DistillationModel requires training_config for batching.")
        batch_sz = self.training_config.batch_size

        def teacher_fn(x: JaxF64) -> JaxF64:
            x_proj = projection_layer(x) if projection_layer is not None else x
            return jax.lax.stop_gradient(self.teacher(x_proj))

        if X.shape[0] > batch_sz:
            targets_np = batched_compute(
                teacher_fn,
                to_np_f64(X),
                batch_size=batch_sz,
                desc="Teacher (Eval)",
                file="model.py",
                step="eval"
            )
            teacher_targets = to_jax_f64(targets_np)
        else:
            teacher_targets = self._compute_teacher_targets(X, projection_layer=projection_layer)

        student_metrics = self.student.evaluate(X, teacher_targets)
        return student_metrics

    def get_topology_meta(self) -> TopologyMeta:
        if self.topology_meta:
            return self.topology_meta
        if self.student.topology_meta:
            return self.student.topology_meta
        return self.teacher.topology_meta

    # ------------------------------------------------------------------ #
    # ClosedLoopGenerativeModel Interface                                #
    # ------------------------------------------------------------------ #

    def initialize_state(self, batch_size: int = 1) -> JaxF64:
        if self._input_dim is None:
            raise RuntimeError("DistillationModel._input_dim not set. Call train() first.")
        return jnp.zeros((batch_size, self._window_size, self._input_dim))

    def step(self, state: JaxF64, inputs: JaxF64) -> tuple[JaxF64, JaxF64]:
        next_state = jnp.concatenate([state[:, 1:], inputs[:, None, :]], axis=1)
        batch_size = next_state.shape[0]
        flat_input = next_state.reshape(batch_size, -1)
        
        from reservoir.models.nn.base import BaseFlaxModel
        output = BaseFlaxModel.predict(self.student, flat_input)
        return next_state, output

    def forward(self, state: JaxF64, input_data: JaxF64) -> tuple[JaxF64, JaxF64]:
        inputs_transposed = jnp.swapaxes(input_data, 0, 1)
        final_state, stacked_outputs = jax.lax.scan(self.step, state, inputs_transposed)
        stacked_outputs = jnp.swapaxes(stacked_outputs, 0, 1)
        return final_state, stacked_outputs
