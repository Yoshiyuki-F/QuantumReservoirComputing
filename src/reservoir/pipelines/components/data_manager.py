from dataclasses import replace
from reservoir.training.presets import get_training_preset, TrainingConfig
from reservoir.utils.reporting import print_feature_stats

from reservoir.core.types import to_jax_f64
from reservoir.pipelines.config import FrontendContext, DatasetMetadata
from reservoir.data.loaders import load_dataset_with_validation_split
from reservoir.data.presets import DATASET_REGISTRY
from reservoir.layers.preprocessing import create_preprocessor, register_preprocessors
from reservoir.layers.projection import (
    create_projection,
    register_projections
)
from reservoir.models.config import (
    PipelineConfig, 
    RandomProjectionConfig, CenterCropProjectionConfig, ResizeProjectionConfig, PolynomialProjectionConfig, PCAProjectionConfig,
    BoundedAffinePCAConfig,
    RawConfig, StandardScalerConfig, MinMaxScalerConfig, AffineScalerConfig, BoundedAffineScalerConfig,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.data.structs import SplitDataset
    from reservoir.data.identifiers import Dataset

# Register projection configs once at module level #TODO there is a interface for config
register_projections(CenterCropProjectionConfig, RandomProjectionConfig, ResizeProjectionConfig, PolynomialProjectionConfig, PCAProjectionConfig, BoundedAffinePCAConfigClass=BoundedAffinePCAConfig)

# Register preprocessor configs #TODO there is a interface for config
register_preprocessors(RawConfig, StandardScalerConfig, MinMaxScalerConfig, AffineScalerConfig, BoundedAffineScalerConfig)


class PipelineDataManager:
    """
    Handles data loading, preprocessing, and projection.
    Encapsulates memory management and stats logging for data preparation.
    """

    def __init__(self, dataset: Dataset, config: PipelineConfig, training_override: TrainingConfig | None = None):
        self.config = config
        self.dataset = dataset
        self.training_override = training_override
        self.metadata: DatasetMetadata | None = None

    def prepare(self) -> FrontendContext:
        """
        Step 1, 2, 3: Load, Preprocess, Project.
        """
        # --- Step 1: Load ---
        self.metadata, raw_split = self._load_dataset()
        
        # --- Step 2 & 3: Frontend Processing ---
        frontend_ctx = self._process_frontend(raw_split)
        
        # Explicitly release raw_split (though extracted method scope handles it mostly)
        del raw_split
        
        return frontend_ctx

    def _load_dataset(self) -> tuple[DatasetMetadata, SplitDataset]:
        """Step 1: Resolve presets and load dataset without mutating inputs later."""
        print("\n[data_manager.py] === Step 1: Loading Dataset ===")
        dataset_enum, dataset_preset = self.dataset, DATASET_REGISTRY.get(self.dataset)
        if dataset_preset is None:
            raise ValueError(f"Unknown dataset: {dataset_enum}. Not found in DATASET_REGISTRY.")

        training_cfg = self.training_override or get_training_preset("standard")
        dataset_split = load_dataset_with_validation_split(
            self.dataset,
            training_cfg,
        )

        self._log_dataset_stats(dataset_split, "1")

        # User Request: Use full 3D shape (Batch, Time, Feature) for topology logging
        input_shape = dataset_split.train_X.shape if dataset_split.train_X is not None else ()

        metadata = DatasetMetadata(
            dataset=dataset_enum,
            dataset_name=dataset_preset.name,
            preset=dataset_preset,
            training=training_cfg,
            classification=dataset_preset.classification,
            input_shape=input_shape,
        )
        return metadata, dataset_split

    def _process_frontend(self, raw_split: SplitDataset) -> FrontendContext:
        """
        Step 2 Apply preprocessing.
        """
        print("\n[datamanager.py] === Step 2: Preprocessing ===")
        preprocessing_config = self.config.preprocess
        if preprocessing_config is not None:
            print(f"[data_manager.py] {preprocessing_config}")
        
        # Factory dispatch on config type
        preprocessor = create_preprocessor(preprocessing_config)

        data_split = raw_split
        train_X = data_split.train_X
        val_X = data_split.val_X
        test_X = data_split.test_X

        # Apply preprocessing pipeline
        from reservoir.layers.preprocessing import IdentityPreprocessor
        if not isinstance(preprocessor, IdentityPreprocessor):
            train_X = preprocessor.fit_transform(train_X)
            if val_X is not None:
                val_X = preprocessor.transform(val_X)
            if test_X is not None:
                test_X = preprocessor.transform(test_X)
                
            # For Regression, targets (y) should also be scaled if they share the domain (Auto-Regression)
            if not (self.metadata.classification if self.metadata else False):
                 # Note: use transform to reuse fitted parameters
                 if data_split.train_y is not None:
                     data_split = replace(data_split, train_y=preprocessor.transform(data_split.train_y))
                 if data_split.val_y is not None:
                     data_split = replace(data_split, val_y=preprocessor.transform(data_split.val_y))
                 if data_split.test_y is not None:
                     data_split = replace(data_split, test_y=preprocessor.transform(data_split.test_y))

        # Re-package for stats logging
        preprocessed_split = replace(data_split, train_X=train_X, val_X=val_X, test_X=test_X)
        self._log_dataset_stats(preprocessed_split, "2")
        
        # Use full 3D shape
        preprocessed_shape = train_X.shape
        input_dim = int(preprocessed_shape[-1])

        projection_config = self.config.projection
        projection_layer = None
        projected_shape = None
        input_dim_for_factory = input_dim

        if projection_config is None:
            print("\n[datamanager.py] === Step 3: Projection (Skipped) ===")
        else:
            print("\n[datamanager.py] === Step 3+5+6: Projection + Model + Feature Extraction (Fused) ===")
            
            # Use Factory pattern (DI)
            projection_layer = create_projection(
                projection_config, 
                input_dim=input_dim
            )
            
            # Fit PCA if applicable
            if hasattr(projection_layer, 'fit'):
                print(f"[datamanager.py] Fitting {type(projection_layer).__name__} on training data...")
                projection_layer.fit(to_jax_f64(train_X))
            
            # DEFERRED PROJECTION
            projected_output_dim = int(projection_layer.output_dim)
            projected_shape = train_X.shape[:-1] + (projected_output_dim,)
            input_dim_for_factory = projected_output_dim

            print(f"[datamanager.py]  Projection will be fused with model forward (saves ~{train_X.shape[0] * train_X.shape[1] * projected_output_dim * 8 / 1e9:.1f} GB RAM)")

        return FrontendContext(
            processed_split=preprocessed_split,
            preprocessor=preprocessor,
            preprocessed_shape=preprocessed_shape,
            projected_shape=projected_shape,
            input_shape_for_meta=projected_shape if projected_shape else preprocessed_shape,
            input_dim_for_factory=input_dim_for_factory,
            projection_layer=projection_layer,
        )

    @staticmethod
    def _log_dataset_stats(dataset: SplitDataset, stage: str):
        """Centralized stats logging."""
        print_feature_stats(dataset.train_X, "data_manager.py", f"{stage}:X:train")
        if dataset.train_y is not None:
             print_feature_stats(dataset.train_y,"data_manager.py",  f"{stage}:y:train")
             
        if dataset.val_X is not None:
            print_feature_stats(dataset.val_X,"data_manager.py",  f"{stage}:X:val")
            if dataset.val_y is not None:
                print_feature_stats(dataset.val_y,"data_manager.py",  f"{stage}:y:val")
                
        if dataset.test_X is not None:
            print_feature_stats(dataset.test_X,"data_manager.py",  f"{stage}:X:test")
            if dataset.test_y is not None:
                print_feature_stats(dataset.test_y,"data_manager.py",  f"{stage}:y:test")
