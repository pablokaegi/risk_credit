"""
Distress Classifier

Random Forest and Gradient Boosting classifiers for financial distress prediction
with TimeSeriesSplit cross-validation and hyperparameter tuning.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

from src.model.preprocessing import (
    FinancialDataPreprocessor, 
    handle_class_imbalance, 
    temporal_train_test_split
)


logger = logging.getLogger(__name__)


class DistressClassifier:
    """
    Financial Distress Classifier with embedded preprocessing and hyperparameter tuning.
    
    Supports Random Forest and Gradient Boosting models with:
        - Automatic class imbalance handling (SMOTE)
        - TimeSeriesSplit cross-validation
        - GridSearchCV hyperparameter optimization
        - Feature importance analysis
    
    Attributes:
        model_type (str): Type of model ('random_forest' or 'gradient_boosting')
        model: Trained classifier
        preprocessor: Fitted preprocessor
        best_params: Best hyperparameters from GridSearchCV
        feature_importance: Feature importance scores
        
    Example:
        >>> clf = DistressClassifier(model_type='random_forest')
        >>> clf.load_data('data/processed/features.csv')
        >>> clf.preprocess()
        >>> clf.train()
        >>> clf.evaluate()
        >>> clf.save_model('models/distress_model_rf.pkl')
    """
    
    RF_PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    GB_PARAM_GRID = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        
        self.model = None
        self.preprocessor = None
        self.best_params = None
        self.feature_importance = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        
        logger.info(f"DistressClassifier initialized with model_type={model_type}")
    
    def load_data(
        self, 
        data_path: Union[str, Path],
        target_column: str = 'target',
        test_size: float = 0.2
    ) -> None:
        """
        Load and split data for training and testing.
        
        Args:
            data_path: Path to CSV file with features and target
            target_column: Name of target column (default: 'target')
            test_size: Fraction of data for test set (default: 0.2)
        """
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Temporal split
        self.X_train, self.X_test, self.y_train, self.y_test = temporal_train_test_split(
            df,
            test_size=test_size,
            target_column=target_column
        )
    
    def preprocess(
        self,
        imputation_strategy: str = 'knn',
        scaling_method: str = 'robust',
        handle_imbalance: bool = True,
        imbalance_method: str = 'smote'
    ) -> None:
        """
        Preprocess data: imputation, scaling, and class imbalance handling.
        
        Args:
            imputation_strategy: Strategy for missing values ('knn' or 'median')
            scaling_method: Scaling method ('robust', 'standard', 'minmax')
            handle_imbalance: Whether to handle class imbalance (default: True)
            imbalance_method: Resampling method ('smote', 'adasyn', 'smote_tomek')
        """
        # Initialize preprocessor
        self.preprocessor = FinancialDataPreprocessor(
            imputation_strategy=imputation_strategy,
            scaling_method=scaling_method
        )
        
        # Fit and transform training data
        self.X_train = pd.DataFrame(
            self.preprocessor.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        # Transform test data (only transform, not fit!)
        self.X_test = pd.DataFrame(
            self.preprocessor.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # Handle class imbalance
        if handle_imbalance:
            self.X_train_resampled, self.y_train_resampled = handle_class_imbalance(
                self.X_train,
                self.y_train,
                method=imbalance_method
            )
        else:
            self.X_train_resampled = self.X_train
            self.y_train_resampled = self.y_train
        
        logger.info("Preprocessing complete")
    
    def train(self, use_grid_search: bool = False, cv_splits: int = 5) -> None:
        """
        Train the classifier.
        
        Args:
            use_grid_search: Whether to perform hyperparameter tuning (default: False)
            cv_splits: Number of cross-validation splits (default: 5)
        """
        # Initialize base model
        if self.model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            param_grid = self.RF_PARAM_GRID
            
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state
            )
            param_grid = self.GB_PARAM_GRID
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Hyperparameter tuning
        if use_grid_search:
            logger.info("Performing GridSearchCV...")
            
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=tscv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_resampled, self.y_train_resampled)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best F1-Score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            logger.info("Training model with default parameters...")
            self.model = base_model
            self.model.fit(self.X_train_resampled, self.y_train_resampled)
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info("Model training complete")
    
    def evaluate(self) -> Dict:
        """
        Evaluate model performance on test set.
        
        Returns:
            Dictionary with evaluation metrics:
                {'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix'}
        """
        # Predict
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(self.y_test, y_pred)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{report}")
        
        return metrics
    
    def cross_validate(self, cv_splits: int = 5) -> Dict:
        """
        Perform cross-validation with TimeSeriesSplit.
        
        Args:
            cv_splits: Number of splits (default: 5)
            
        Returns:
            Dictionary with cross-validation results
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train_resampled)):
            X_train_fold = self.X_train_resampled.iloc[train_idx]
            y_train_fold = self.y_train_resampled.iloc[train_idx]
            X_val_fold = self.X_train_resampled.iloc[val_idx]
            y_val_fold = self.y_train_resampled.iloc[val_idx]
            
            # Train on fold
            self.model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation
            y_pred = self.model.predict(X_val_fold)
            
            # Metrics
            precision = precision_score(y_val_fold, y_pred)
            recall = recall_score(y_val_fold, y_pred)
            f1 = f1_score(y_val_fold, y_pred)
            
            fold_results.append({
                'fold': fold + 1,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            logger.info(
                f"Fold {fold + 1}: Precision={precision:.3f}, "
                f"Recall={recall:.3f}, F1={f1:.3f}"
            )
        
        # Aggregate results
        df_results = pd.DataFrame(fold_results)
        
        metrics = {
            'mean_precision': df_results['precision'].mean(),
            'mean_recall': df_results['recall'].mean(),
            'mean_f1': df_results['f1'].mean(),
            'std_precision': df_results['precision'].std(),
            'std_recall': df_results['recall'].std(),
            'std_f1': df_results['f1'].std(),
            'fold_results': fold_results
        }
        
        logger.info(f"Mean F1-Score: {metrics['mean_f1']:.4f} ± {metrics['std_f1']:.4f}")
        
        return metrics
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model and preprocessor to file.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'DistressClassifier':
        """
        Load trained model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            DistressClassifier instance with loaded model
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.preprocessor = model_data['preprocessor']
        instance.best_params = model_data['best_params']
        instance.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance


def main():
    """Command-line interface for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train financial distress classifier')
    parser.add_argument('--input', required=True, help='Input CSV file with features')
    parser.add_argument('--output', default='models/distress_model.pkl', help='Output model file')
    parser.add_argument('--model-type', default='random_forest', choices=['random_forest', 'gradient_boosting'])
    parser.add_argument('--grid-search', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of CV splits')
    
    args = parser.parse_args()
    
    # Initialize classifier
    clf = DistressClassifier(model_type=args.model_type)
    
    # Load data
    clf.load_data(args.input)
    
    # Preprocess
    clf.preprocess(handle_imbalance=True)
    
    # Train
    clf.train(use_grid_search=args.grid_search, cv_splits=args.cv_splits)
    
    # Evaluate
    metrics = clf.evaluate()
    
    # Save model
    clf.save_model(args.output)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())