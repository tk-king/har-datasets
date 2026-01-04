from typing import Any, List

from whar_datasets.processing.steps.processing_step import ProcessingStep


class ProcessingPipeline:
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps

    def run(self, force_recompute: bool | List[bool] | None = None) -> Any:
        if isinstance(force_recompute, list):
            assert len(self.steps) == len(force_recompute)
            for step, fr in zip(self.steps, force_recompute):
                step.run(fr)
        elif isinstance(force_recompute, bool):
            for step in self.steps:
                step.run(force_recompute)
