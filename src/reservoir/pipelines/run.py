"""
pipelines/run.py
Unified Pipeline Runner (Refactored)

Architecture V2 Compliance:
- Orchestration Only: No low-level logic (loops, math, printing) allowed.
- Component-Based: Delegates to DataManager, ModelBuilder, Executor, and Reporter.
"""


# Refactored Components
from reservoir.pipelines.components.data_manager import PipelineDataManager
from reservoir.pipelines.components.model_builder import PipelineModelBuilder
from reservoir.pipelines.components.executor import PipelineExecutor
from reservoir.pipelines.components.reporter import ResultReporter
from reservoir.pipelines.components.data_coordinator import DataCoordinator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.training.presets import TrainingConfig
    from reservoir.models.presets import PipelineConfig
    from reservoir.core.types import ConfigDict
    from reservoir.data.identifiers import Dataset


def run_pipeline(
    config: PipelineConfig, 
    dataset: Dataset, 
    training_config: TrainingConfig | None = None
) -> ConfigDict:
    """
    Declarative orchestrator for the unified pipeline.
    
    Flow:
    1. Data Preparation (Load -> Preprocess -> Project)
    2. Model Construction (Factory -> Topology)
    3. Execution (Train -> Extract Features -> Readout Fit)
    4. Reporting (Metrics -> Logs -> Disk)
    """
    
    # === Step 1-3: Data Preparation ===
    # DataManager encapsulates loading, splitting, scaling, and projection logic.
    # It handles memory management and stats logging internally.
    data_mgr = PipelineDataManager(dataset, config, training_config)
    frontend_ctx = data_mgr.prepare()
    if data_mgr.metadata is None:
        raise ValueError("DataManager.prepare() must set metadata")
    metadata = data_mgr.metadata

    # === Step 3.5: Model Stack Construction ===
    # ModelBuilder encapsulates ModelFactory and ReadoutFactory calls.
    # It computes topology metadata automatically.
    stack = PipelineModelBuilder.build(config, frontend_ctx, metadata)

    # === Step 4,5,6,7: Execution ===
    # Executor handles the training loop, feature extraction (batched),
    # strategy selection (Classification vs ClosedLoop), and fitting.
    coordinator = DataCoordinator(frontend_ctx, metadata)
    executor = PipelineExecutor(stack, frontend_ctx, metadata, coordinator)
    execution_results = executor.run(config)

    # === Step 8: Reporting ===
    # Reporter handles result formatting, computing final scores, 
    # and generating the HTML/JSON report.
    reporter = ResultReporter(stack, frontend_ctx, metadata)
    final_results = reporter.compile_and_save(execution_results, config)

    from typing import cast
    return cast("ConfigDict", final_results)
