"""Predictor module for production deployment."""
import logging
import pickle
from pathlib import Path
from typing import Dict, Union, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DistressPredictor:
    """Production predictor for financial distress classification."""
    
    def __init__(self, model_path: Union[str, Path]):
        """Initialize predictor with trained model."""
        self.model_path = Path(model_path)
        self.model = None
        self.preprocessor = None
        self._load_model()
    
    def _load_model(self):
        """Load model and preprocessor from file."""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        logger.info(f"Model loaded from {self.model_path}")
    
    def predict(self, **kwargs) -> Dict:
        """
        Predict distress for single company.
        
        Args:
            **kwargs: Financial ratios as keyword arguments
            
        Returns:
            Dictionary with 'probability' and 'class'
        """
        # Create DataFrame from kwargs
        df = pd.DataFrame([kwargs])
        
        # Preprocess
        df_processed = self.preprocessor.transform(df)
        
        # Predict
        probability = self.model.predict_proba(df_processed)[0, 1]
        predicted_class = 'Distress' if probability > 0.5 else 'Healthy'
        
        return {'probability': probability, 'class': predicted_class}
    
    def predict_batch(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Predict distress for batch of companies.
        
        Args:
            filepath: Path to CSV with financial ratios
            
        Returns:
            DataFrame with predictions
        """
        df = pd.read_csv(filepath)
        df_processed = self.preprocessor.transform(df)
        
        probabilities = self.model.predict_proba(df_processed)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        df['distress_probability'] = probabilities
        df['prediction'] = ['Distress' if p == 1 else 'Healthy' for p in predictions]
        
        return df
    
    def set_threshold(self, threshold: float):
        """Set prediction threshold."""
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model file path')
    parser.add_argument('--input', required=True, help='Input CSV')
    parser.add_argument('--output', help='Output CSV')
    args = parser.parse_args()
    
    predictor = DistressPredictor(args.model)
    predictions = predictor.predict_batch(args.input)
    
    output_path = args.output or 'predictions.csv'
    predictions.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")