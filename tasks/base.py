from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import jinja2
from dataclasses import dataclass

from dataloaders.base import HuggingFaceDataset


@dataclass
class TaskConfig:
    name: str
    dataset_path: str
    dataset_name: str
    training_split: str
    test_split: str
    system_prompt: str
    ground_truth_format: str
    few_shot_samples: Optional[List[Dict[str, Any]]] = None
    stop: Optional[str] = None
    # limit the number of examples to load from each split (None==all)
    max_train_examples: Optional[int] = None
    max_test_examples: Optional[int] = None
    # optional filters (only used by datasets that include these fields)
    allowed_levels: Optional[List[str]] = None
    allowed_types: Optional[List[str]] = None


class Task(ABC):
    def __init__(self, config: TaskConfig):
        self.config = config
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from HuggingFace and apply any size limits."""
        self.train_dataset = HuggingFaceDataset(
            self.config.dataset_path,
            self.config.dataset_name,
            split=self.config.training_split
        ).load()
        self.test_dataset = HuggingFaceDataset(
            self.config.dataset_path,
            self.config.dataset_name,
            split=self.config.test_split
        ).load()

        # apply optional filters (level/type) before sampling
        if self.config.allowed_levels is not None:
            self.train_dataset = [ex for ex in self.train_dataset if ex.get("level") in self.config.allowed_levels]
            self.test_dataset = [ex for ex in self.test_dataset if ex.get("level") in self.config.allowed_levels]
        if self.config.allowed_types is not None:
            self.train_dataset = [ex for ex in self.train_dataset if ex.get("type") in self.config.allowed_types]
            self.test_dataset = [ex for ex in self.test_dataset if ex.get("type") in self.config.allowed_types]

        # apply optional limits
        # use a fixed seed for reproducible sampling
        if self.config.max_train_examples is not None:
            import random
            random.seed(42)
            self.train_dataset = random.sample(
                self.train_dataset,
                min(self.config.max_train_examples, len(self.train_dataset))
            )
        if self.config.max_test_examples is not None:
            import random
            random.seed(42)
            self.test_dataset = random.sample(
                self.test_dataset,
                min(self.config.max_test_examples, len(self.test_dataset))
            )

    def get_system_prompt(self, example: Dict[str, Any]) -> str:
        """Render system prompt with example variables"""
        template = jinja2.Template(self.config.system_prompt)
        return template.render(**example)

    def format_ground_truth(self, example: Dict[str, Any]) -> str:
        """Format ground truth using the template"""
        template = jinja2.Template(self.config.ground_truth_format)
        return template.render(**example)

    def get_test_examples(self) -> List[Dict[str, Any]]:
        """Get test examples from the dataset"""
        return [dict(example) for example in self.test_dataset]

    def get_train_examples(self) -> List[Dict[str, Any]]:
        """Get training examples from the dataset"""
        return [dict(example) for example in self.train_dataset]

    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Parse model response into expected format"""
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
        """Compute metrics for the task"""
        pass
