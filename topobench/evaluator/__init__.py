"""Evaluators for model evaluation."""

from torchmetrics.classification import (
    AUROC,
    ROC,
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from .metrics import ExampleRegressionMetric

# Define metrics
METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "auroc": AUROC,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "example": ExampleRegressionMetric,
    "roc": ROC,
    "f1": F1Score,
}

from .base import AbstractEvaluator  # noqa: E402
from .evaluator import TBEvaluator  # noqa: E402

__all__ = [
    "METRICS",
    "AbstractEvaluator",
    "TBEvaluator",
]
