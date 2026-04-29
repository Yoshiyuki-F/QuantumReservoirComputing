
from typing import TYPE_CHECKING

from reservoir.models import ModelFactory
from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
from reservoir.readout.factory import ReadoutFactory

if TYPE_CHECKING:
    from reservoir.layers.adapters import Adapter
    from reservoir.models.presets import PipelineConfig
    from reservoir.core.types import DetailsMeta, ShapesMeta, TopologyMeta


class PipelineModelBuilder:
    """
    Handles Model and Readout construction.
    Computes and populates topology metadata.
    """

    @staticmethod
    def build(config: PipelineConfig, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata) -> ModelStack:
        """Step 4: Instantiate model + readout and enrich topology metadata."""
        
        # 1. Infer Output Dimension
        processed = frontend_ctx.processed_split
        if dataset_meta.preset.config.n_output is not None:
            meta_n_outputs = int(dataset_meta.preset.config.n_output)
        else:
            target_sample = processed.train_y if processed.train_y is not None else processed.test_y
            if target_sample is None:
                raise ValueError("Unable to infer output dimension without targets.")
            meta_n_outputs = (
                int(target_sample.shape[-1]) if hasattr(target_sample, "shape") and len(target_sample.shape) > 1 else 1
            )

        # 2. Identify Input Shape
        # input_shape_for_meta is already computed in DataManager (projected or preprocessed shape)
        
        # 3. Create Model
        model = ModelFactory.create_model(
            config=config,
            training=dataset_meta.training,
            input_dim=frontend_ctx.input_dim_for_factory,
            output_dim=meta_n_outputs,
            input_shape=frontend_ctx.input_shape_for_meta,
            classification=dataset_meta.classification,
        )

        # 4. Build Topology Metadata
        topo_meta: TopologyMeta = model.get_topology_meta()

        shapes_meta: ShapesMeta
        existing_shapes = topo_meta.get("shapes")
        if existing_shapes is None:
            shapes_meta = {}
        else:
            shapes_meta = existing_shapes
        shapes_meta["input"] = dataset_meta.input_shape
        shapes_meta["preprocessed"] = frontend_ctx.preprocessed_shape
        shapes_meta["projected"] = frontend_ctx.projected_shape
        if "output" not in shapes_meta:
            shapes_meta["output"] = (meta_n_outputs,)
        topo_meta["shapes"] = shapes_meta

        details_meta: DetailsMeta
        existing_details = topo_meta.get("details")
        if existing_details is None:
            details_meta = {}
        else:
            details_meta = existing_details
        details_meta["preprocess"] = type(frontend_ctx.preprocessor).__name__ if frontend_ctx.preprocessor else None
        topo_meta["details"] = details_meta
        
        # 5. Create Readout
        is_classification = dataset_meta.classification
        readout = ReadoutFactory.create_readout(config.readout, is_classification, dataset_meta.training)
        
        if readout is not None:
            readout_name = type(readout).__name__
            if hasattr(readout, 'hidden_layers') and readout.hidden_layers:
                readout_name += f" ({'-'.join(str(v) for v in readout.hidden_layers)})"
            details_meta["readout"] = readout_name
        else:
            details_meta["readout"] = None
        
        metric = "accuracy" if dataset_meta.classification else "mse"
        
        return ModelStack(
            model=model,
            readout=readout,
            topo_meta=topo_meta,
            metric=metric,
            model_label=config.model_type.value.replace("-", "_"),
        )

    @staticmethod
    def get_adapter(stack: ModelStack) -> Adapter | None:
        """Helper to retrieve adapter from various model wrappers in the stack."""
        model = stack.model
        if hasattr(model, 'adapter') and not hasattr(model, 'student'):
            return model.adapter
        return None
