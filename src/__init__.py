"""
Argentine Financial Distress Prediction Model

This package provides a production-ready machine learning pipeline for predicting
financial distress in Argentine publicly traded companies using regulatory data
from the Comisión Nacional de Valores (CNV).

Modules:
    - data_acquisition: Web scraping and data collection from CNV and BYMA
    - features: Financial ratio calculation and temporal feature engineering
    - model: Preprocessing, model training, and prediction
    - evaluation: Model evaluation metrics and visualization
    - utils: Configuration, logging, and helper functions
"""

from importlib import import_module
from typing import Any

__version__ = "1.0.0"
__author__ = "Pablo Kaegi"
__email__ = "pablokaegi@email.com"

__all__ = [
    "CNVDataExtractor",
    "FinancialRatioEngine", 
    "DistressClassifier",
    "DistressPredictor",
    "ModelEvaluator",
]


_EXPORTS = {
    "CNVDataExtractor": ("src.data_acquisition.cnv_scraper", "CNVDataExtractor"),
    "FinancialRatioEngine": ("src.features", "FinancialRatioEngine"),
    "DistressClassifier": ("src.model.classifier", "DistressClassifier"),
    "DistressPredictor": ("src.model.predictor", "DistressPredictor"),
    "ModelEvaluator": ("src.evaluation", "ModelEvaluator"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src' has no attribute '{name}'")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value