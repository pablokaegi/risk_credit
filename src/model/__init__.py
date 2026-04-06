"""
Model Training and Prediction Module

This module implements classification models for financial distress prediction,
including Random Forest and Gradient Boosting with hyperparameter tuning.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "DistressClassifier",
    "DistressPredictor",
    "FinancialDataPreprocessor",
    "handle_class_imbalance",
    "temporal_train_test_split",
    "create_preprocessing_pipeline",
    "create_time_series_cv",
]


_EXPORTS = {
    "DistressClassifier": ("src.model.classifier", "DistressClassifier"),
    "DistressPredictor": ("src.model.predictor", "DistressPredictor"),
    "FinancialDataPreprocessor": ("src.model.preprocessing", "FinancialDataPreprocessor"),
    "handle_class_imbalance": ("src.model.preprocessing", "handle_class_imbalance"),
    "temporal_train_test_split": ("src.model.preprocessing", "temporal_train_test_split"),
    "create_preprocessing_pipeline": ("src.model.preprocessing", "create_preprocessing_pipeline"),
    "create_time_series_cv": ("src.model.preprocessing", "create_time_series_cv"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.model' has no attribute '{name}'")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value