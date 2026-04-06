"""
Model Preprocessing Module

This module implements preprocessing pipelines for financial distress prediction,
including imputation, scaling, outlier treatment, and class imbalance handling.

Key Components:
    - KNN Imputation for missing values
    - Robust Scaling for heavy-tailed distributions
    - SMOTE for class imbalance
    - TimeSeriesSplit for temporal cross-validation

Author: Pablo Kaegi
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


logger = logging.getLogger(__name__)


class FinancialDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for financial data preprocessing.
    
    Handles:
        - Missing value imputation (KNN with financial logic)
        - Outlier treatment (winsorization at 1st/99th percentiles)
        - Feature scaling (RobustScaler for heavy-tailed distributions)
    
    Attributes:
        imputation_strategy (str): Strategy for handling missing values ('knn', 'median')
        outlier_method (str): Method for outlier treatment ('winsorize', 'clip', 'log_transform')
        imputer: Fitted imputer
        scaler: Fitted scaler
        
    Example:
        >>> preprocessor = FinancialDataPreprocessor()
        >>> X_train_processed = preprocessor.fit_transform(X_train)
        >>> X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        imputation_strategy: str = 'knn',
        outlier_method: str = 'winsorize',
        knn_neighbors: int = 5,
        scaling_method: str = 'robust'
    ):
        """
        Initialize the preprocessor.
        
        Args:
            imputation_strategy: Strategy for missing values ('knn' or 'median')
            outlier_method: Method for outlier treatment ('winsorize', 'clip', 'log_transform')
            knn_neighbors: Number of neighbors for KNN imputation (default: 5)
            scaling_method: Scaling method ('robust', 'standard', 'minmax')
        """
        self.imputation_strategy = imputation_strategy
        self.outlier_method = outlier_method
        self.knn_neighbors = knn_neighbors
        self.scaling_method = scaling_method
        
        self.imputer = None
        self.scaler = None
        self.outlier_bounds_ = None
        
        logger.info(
            f"FinancialDataPreprocessor initialized with "
            f"imputation={imputation_strategy}, outlier={outlier_method}, "
            f"scaling={scaling_method}"
        )
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn imputation, scaling, and outlier parameters from training data.
        
        Args:
            X: Training features DataFrame
            y: Target variable (optional, not used in fitting)
            
        Returns:
            self (fitted preprocessor)
        """
        logger.info("Fitting preprocessor...")
        
        # Handle missing values (imputation)
        self._fit_imputer(X)
        
        # Handle outliers
        self._fit_outlier_bounds(X)
        
        # Scale features
        self._fit_scaler(X)
        
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def _fit_imputer(self, X: pd.DataFrame) -> None:
        """
        Fit imputer on training data.
        
        Args:
            X: Training features DataFrame
        """
        if self.imputation_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self.imputer.fit(X)
            logger.info(f"KNN Imputer fitted with {self.knn_neighbors} neighbors")
            
        elif self.imputation_strategy == 'median':
            # Median imputation (simple and robust to outliers)
            self.imputer = X.median()
            logger.info("Median Imputer fitted")
            
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")
    
    def _fit_outlier_bounds(self, X: pd.DataFrame) -> None:
        """
        Learn outlier bounds for winsorization.
        
        Args:
            X: Training features DataFrame
        """
        if self.outlier_method == 'winsorize':
            # Store 1st and 99th percentiles for winsorization
            self.outlier_bounds_ = {
                'lower': X.quantile(0.01),
                'upper': X.quantile(0.99)
            }
            logger.info("Outlier bounds computed for winsorization (1st/99th percentiles)")
            
        elif self.outlier_method == 'clip':
            # Clip at min/max values
            self.outlier_bounds_ = {
                'lower': X.min(),
                'upper': X.max()
            }
            
        elif self.outlier_method == 'log_transform':
            # No bounds needed for log transform
            self.outlier_bounds_ = None
            
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
    
    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """
        Fit feature scaler on training data.
        
        Args:
            X: Training features DataFrame
        """
        if self.scaling_method == 'robust':
            self.scaler = RobustScaler()  # Uses median and IQR (robust to outliers)
            self.scaler.fit(X)
            logger.info("RobustScaler fitted")
            
        elif self.scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            logger.info("StandardScaler fitted")
            
        elif self.scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
            logger.info("MinMaxScaler fitted")
            
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to data.
        
        Args:
            X: Features DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming data...")
        
        X_transformed = X.copy()
        
        # Step 1: Handle missing values
        X_transformed = self._apply_imputation(X_transformed)
        
        # Step 2: Handle outliers
        X_transformed = self._apply_outlier_treatment(X_transformed)
        
        # Step 3: Scale features
        X_transformed = self._apply_scaling(X_transformed)
        
        logger.info(f"Transformation complete. Shape: {X_transformed.shape}")
        
        return X_transformed
    
    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to handle missing values.
        
        Args:
            X: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        if self.imputation_strategy == 'knn':
            # KNN imputation
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )
            return X_imputed
            
        elif self.imputation_strategy == 'median':
            # Median imputation
            return X.fillna(self.imputer)
        
        else:
            return X
    
    def _apply_outlier_treatment(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier treatment.
        
        Args:
            X: DataFrame with potential outliers
            
        Returns:
            DataFrame with treated outliers
        """
        if self.outlier_method == 'winsorize':
            # Clip at 1st/99th percentiles
            return X.clip(
                lower=self.outlier_bounds_['lower'],
                upper=self.outlier_bounds_['upper'],
                axis=1
            )
            
        elif self.outlier_method == 'clip':
            # Clip at min/max
            return X.clip(
                lower=self.outlier_bounds_['lower'],
                upper=self.outlier_bounds_['upper'],
                axis=1
            )
            
        elif self.outlier_method == 'log_transform':
            # Apply log transform (add small constant to avoid log(0))
            return np.log1p(X.abs())  # Use abs() to handle negative values
            
        else:
            return X
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature scaling.
        
        Args:
            X: DataFrame to scale
            
        Returns:
            Scaled DataFrame
        """
        if self.scaler is None:
            return X
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled


def create_preprocessing_pipeline(
    imputation_strategy: str = 'knn',
    outlier_method: str = 'winsorize',
    scaling_method: str = 'robust',
    knn_neighbors: int = 5
) -> Pipeline:
    """
    Construct sklearn Pipeline for preprocessing.
    
    The pipeline is structured to:
        1. Impute missing values
        2. Handle outliers
        3. Scale features
        4. Resample classes (SMOTETomek for balance)
    
    Args:
        imputation_strategy: Strategy for missing values ('knn' or 'median')
        outlier_method: Method for outlier treatment ('winsorize', 'clip', 'log_transform')
        scaling_method: Scaling method ('robust', 'standard', 'minmax')
        knn_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        sklearn Pipeline object
        
    Example:
        >>> pipeline = create_preprocessing_pipeline()
        >>> X_train_processed = pipeline.fit_transform(X_train, y_train)
    """
    preprocessor = FinancialDataPreprocessor(
        imputation_strategy=imputation_strategy,
        outlier_method=outlier_method,
        knn_neighbors=knn_neighbors,
        scaling_method=scaling_method
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    logger.info("Preprocessing pipeline created")
    
    return pipeline


def handle_class_imbalance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'smote',
    sampling_strategy: str = 'auto',
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using resampling techniques.
    
    Args:
        X: Features DataFrame
        y: Target Series
        method: Resampling method ('smote', 'adasyn', 'undersample', 'smote_tomek')
        sampling_strategy: Sampling strategy ('auto', 'minority', or float)
        k_neighbors: Number of neighbors for SMOTE/ADASYN
        
    Returns:
        Resampled (X, y) tuple
        
    Example:
        >>> X_resampled, y_resampled = handle_class_imbalance(X_train, y_train, method='smote')
    """
    logger.info(f"Handling class imbalance using {method}...")
    logger.info(f"Original distribution: {y.value_counts().to_dict()}")
    
    if method == 'smote':
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42
        )
        
    elif method == 'adasyn':
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=k_neighbors,
            random_state=42
        )
        
    elif method == 'undersample':
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=42
        )
        
    elif method == 'smote_tomek':
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=42
        )
        
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    logger.info(f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    logger.info(f"Shape: {X_resampled.shape}")
    
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_column: str = 'period',
    target_column: str = 'target'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically to prevent data leakage.
    
    Training data always precedes test data chronologically.
    This is critical for financial time series where future
    information must not leak into the training set.
    
    Args:
        df: DataFrame with features and target
        test_size: Fraction of data for test set (default: 0.2)
        date_column: Name of date/period column
        target_column: Name of target column
        
    Returns:
        (X_train, X_test, y_train, y_test) tuple
        
    Example:
        >>> X_train, X_test, y_train, y_test = temporal_train_test_split(df, test_size=0.2)
    """
    logger.info("Performing temporal train/test split...")
    
    # Sort by date
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    
    # Determine split point
    n_total = len(df_sorted)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    # Split chronologically
    train_df = df_sorted.iloc[:n_train]
    test_df = df_sorted.iloc[n_train:]
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in [date_column, target_column, 'ticker']]
    
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df[target_column]
    y_test = test_df[target_column]
    
    logger.info(f"Train set: {len(X_train)} samples ({train_df[date_column].min()} to {train_df[date_column].max()})")
    logger.info(f"Test set: {len(X_test)} samples ({test_df[date_column].min()} to {test_df[date_column].max()})")
    
    # Check for temporal overlap
    train_max_date = train_df[date_column].max()
    test_min_date = test_df[date_column].min()
    
    if train_max_date >= test_min_date:
        logger.warning(f"Temporal overlap detected: train ends {train_max_date}, test starts {test_min_date}")
    
    return X_train, X_test, y_train, y_test


def create_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """
    Create TimeSeriesSplit cross-validator for temporal data.
    
    TimeSeriesSplit ensures:
        - Training folds always precede validation folds
        - No future information leaks into past training
        - Realistic out-of-sample performance estimates
    
    Args:
        n_splits: Number of splits (default: 5)
        
    Returns:
        TimeSeriesSplit object
        
    Example:
        >>> tscv = create_time_series_cv(n_splits=5)
        >>> for train_idx, test_idx in tscv.split(X):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    logger.info(f"TimeSeriesSplit created with {n_splits} splits")
    return tscv


def main():
    """Command-line interface for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess financial data')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', default='data/processed/preprocessed.csv', help='Output CSV file')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--imputation', default='knn', choices=['knn', 'median'], help='Imputation strategy')
    parser.add_argument('--scaling', default='robust', choices=['robust', 'standard', 'minmax'], help='Scaling method')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} samples from {args.input}")
    
    # Temporal split
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df,
        test_size=args.test_size,
        target_column=args.target
    )
    
    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline(
        imputation_strategy=args.imputation,
        scaling_method=args.scaling
    )
    
    # Fit and transform
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Save processed data
    train_df = pd.concat([X_train_processed, y_train], axis=1)
    test_df = pd.concat([X_test_processed, y_test], axis=1)
    
    train_path = args.output.replace('.csv', '_train.csv')
    test_path = args.output.replace('.csv', '_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Processed data saved to {train_path} and {test_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())