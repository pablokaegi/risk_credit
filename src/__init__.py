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

__version__ = "1.0.0"
__author__ = "Pablo Kaegi"
__email__ = "pablokaegi@email.com"

# Lazy imports to avoid circular dependencies
__all__ = [
    "CNVDataExtractor",
    "FinancialRatioEngine", 
    "DistressClassifier",
    "DistressPredictor",
    "ModelEvaluator",
]