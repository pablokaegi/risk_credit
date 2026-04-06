"""
Financial Ratio Calculator

This module calculates critical financial ratios from balance sheet and income statement data
for Argentine publicly traded companies. Ratios are computed using vectorized operations
for efficiency and include division-by-zero protection and automatic outlier treatment.

Key Features:
    - Liquidity ratios (Current, Quick, Cash)
    - Leverage ratios (Debt-to-Equity, Debt-to-Assets, Interest Coverage)
    - Profitability ratios (ROA, ROE, Operating Margin)
    - Efficiency ratios (Asset Turnover)
    - Temporal features (Momentum, Trend, Volatility)
    - Sector-relative adjustments

Author: Pablo Kaegi
Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


class FinancialRatioEngine:
    """
    Compute financial ratios from raw balance sheet and income statement data.
    
    Ratios are calculated using vectorized operations for efficiency.
    Missing values are handled according to accounting principles.
    Extreme values are winsorized to prevent undue influence in ML models.
    
    Attributes:
        RATIO_DEFINITIONS (dict): Dictionary mapping ratio names to formulas
        winsorize_limits (tuple): Percentiles for winsorization (default: (0.01, 0.99))
    
    Example:
        >>> engine = FinancialRatioEngine()
        >>> df = engine.load_data('data/raw/cnv_statements/')
        >>> df_with_ratios = engine.compute_all_ratios(df)
        >>> df_final = engine.add_temporal_features(df_with_ratios)
    """
    
    RATIO_DEFINITIONS = {
        'current_ratio': {
            'category': 'liquidity',
            'interpretation': 'Higher is better, typically >1.5 for healthy companies',
            'typical_range': (0.5, 3.0)
        },
        'quick_ratio': {
            'category': 'liquidity',
            'interpretation': 'Excludes illiquid inventory. Critical for companies with slow-moving stock.',
            'typical_range': (0.3, 2.5)
        },
        'cash_ratio': {
            'category': 'liquidity',
            'interpretation': 'Most conservative liquidity measure. Survives in cash crunch scenarios.',
            'typical_range': (0.05, 1.5)
        },
        'debt_to_equity': {
            'category': 'leverage',
            'interpretation': 'Lower is better, typically <2.0 for stable companies',
            'typical_range': (0.0, 5.0)
        },
        'debt_to_assets': {
            'category': 'leverage',
            'interpretation': 'Proportion of assets financed by debt. Should be <0.6 for stability.',
            'typical_range': (0.1, 0.8)
        },
        'interest_coverage': {
            'category': 'leverage',
            'interpretation': 'Ability to service debt. Should be >3.0 for comfortable coverage.',
            'typical_range': (0.5, 20.0)
        },
        'roa': {
            'category': 'profitability',
            'interpretation': 'Asset efficiency. Healthy companies typically have >5%.',
            'typical_range': (-0.5, 0.2)
        },
        'roe': {
            'category': 'profitability',
            'interpretation': 'Returns to shareholders. Higher is better, but >50% may indicate leverage.',
            'typical_range': (-1.0, 0.5)
        },
        'operating_margin': {
            'category': 'profitability',
            'interpretation': 'Core business profitability. Should be positive for sustainability.',
            'typical_range': (-0.5, 0.3)
        },
        'asset_turnover': {
            'category': 'efficiency',
            'interpretation': 'How efficiently assets generate sales. Higher is better.',
            'typical_range': (0.1, 2.0)
        }
    }
    
    def __init__(self, winsorize_limits: Tuple[float, float] = (0.01, 0.99)):
        """
        Initialize the ratio calculation engine.
        
        Args:
            winsorize_limits: Percentiles for winsorization (default: 1st and 99th percentiles)
        """
        self.winsorize_limits = winsorize_limits
        logger.info(f"FinancialRatioEngine initialized with winsorize_limits={winsorize_limits}")
        
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load financial statement data from directory or file.
        
        Args:
            data_path: Path to directory with raw data or CSV file
            
        Returns:
            DataFrame with financial statement data
            
        Raises:
            FileNotFoundError: If data_path does not exist
        """
        data_path = Path(data_path)
        
        if data_path.is_file() and data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}: {len(df)} rows")
            return df
            
        elif data_path.is_dir():
            # Load all CSV files from directory
            all_files = list(data_path.glob('*.csv'))
            if not all_files:
                raise FileNotFoundError(f"No CSV files found in {data_path}")
            
            dfs = [pd.read_csv(f) for f in all_files]
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(all_files)} files from {data_path}: {len(df)} rows")
            return df
            
        else:
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    def compute_all_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all defined financial ratios and add them as columns to DataFrame.
        
        Ratios computed:
            1. Current Ratio = Current Assets / Current Liabilities
            2. Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            3. Cash Ratio = Cash & Equivalents / Current Liabilities
            4. Debt-to-Equity Ratio = Total Liabilities / Total Equity
            5. Debt-to-Assets Ratio = Total Liabilities / Total Assets
            6. Interest Coverage = EBIT / Interest Expense
            7. ROA = Net Income / Total Assets
            8. ROE = Net Income / Total Equity
            9. Operating Margin = Operating Income / Revenue
            10. Asset Turnover = Revenue / Total Assets
        
        Args:
            df: DataFrame with financial statement data (must have 'value' and 'account' columns)
            
        Returns:
            DataFrame with original columns + ratio columns
            
        Example:
            >>> df = pd.DataFrame({
            ...     'ticker': ['GGAL', 'GGAL'],
            ...     'period': ['2024Q1', '2024Q1'],
            ...     'account': ['Current Assets', 'Current Liabilities'],
            ...     'value': [1000000, 500000]
            ... })
            >>> df_ratios = engine.compute_all_ratios(df)
            >>> print('current_ratio' in df_ratios.columns)
            True
        """
        logger.info("Computing all financial ratios...")
        
        # Pivot data to wide format (one row per company-period)
        df_wide = self._pivot_to_wide(df)
        
        # Compute each ratio
        df_wide['current_ratio'] = self._compute_current_ratio(df_wide)
        df_wide['quick_ratio'] = self._compute_quick_ratio(df_wide)
        df_wide['cash_ratio'] = self._compute_cash_ratio(df_wide)
        df_wide['debt_to_equity'] = self._compute_debt_to_equity(df_wide)
        df_wide['debt_to_assets'] = self._compute_debt_to_assets(df_wide)
        df_wide['interest_coverage'] = self._compute_interest_coverage(df_wide)
        df_wide['roa'] = self._compute_roa(df_wide)
        df_wide['roe'] = self._compute_roe(df_wide)
        df_wide['operating_margin'] = self._compute_operating_margin(df_wide)
        df_wide['asset_turnover'] = self._compute_asset_turnover(df_wide)
        
        # Handle outliers
        df_wide = self._handle_outliers(df_wide)
        
        logger.info(f"Computed 10 financial ratios for {len(df_wide)} company-period combinations")
        
        return df_wide
    
    def _pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot long-format financial data to wide format (one row per company-period).
        
        Args:
            df: Long-format DataFrame with columns: ticker, period, account, value
            
        Returns:
            Wide-format DataFrame with one row per company-period
        """
        # Check required columns
        required_cols = ['ticker', 'period', 'account', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Pivot to wide format
        df_wide = df.pivot_table(
            index=['ticker', 'period'],
            columns='account',
            values='value',
            aggfunc='first'  # Take first value if duplicates
        ).reset_index()
        
        # Flatten column names
        df_wide.columns.name = None
        
        logger.debug(f"Pivoted data from {len(df)} rows to {len(df_wide)} rows")
        
        return df_wide
    
    def _safe_divide(
        self, 
        numerator: pd.Series, 
        denominator: pd.Series,
        fill_value: Optional[float] = None
    ) -> pd.Series:
        """
        Division with protection against division by zero.
        
        Args:
            numerator: Series with numerator values
            denominator: Series with denominator values
            fill_value: Value to use when denominator is zero (default: None for NaN)
            
        Returns:
            Series with division result (NaN or fill_value where denominator is zero)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            
            # Replace inf with NaN (arises from non-zero / zero)
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # Fill with specified value if provided
            if fill_value is not None:
                result = result.fillna(fill_value)
        
        return result
    
    def _compute_current_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Current Ratio = Current Assets / Current Liabilities.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Current Ratio values
        """
        current_assets = df.get('Current Assets', df.get('Activo Corriente', pd.Series(index=df.index, dtype=float)))
        current_liabilities = df.get('Current Liabilities', df.get('Pasivo Corriente', pd.Series(index=df.index, dtype=float)))
        
        return self._safe_divide(current_assets, current_liabilities)
    
    def _compute_quick_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Quick Ratio = (Current Assets - Inventory) / Current Liabilities.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Quick Ratio values
        """
        current_assets = df.get('Current Assets', df.get('Activo Corriente', pd.Series(index=df.index, dtype=float)))
        inventory = df.get('Inventory', df.get('Inventarios', pd.Series(0, index=df.index)))
        current_liabilities = df.get('Current Liabilities', df.get('Pasivo Corriente', pd.Series(index=df.index, dtype=float)))
        
        liquid_assets = current_assets - inventory
        return self._safe_divide(liquid_assets, current_liabilities)
    
    def _compute_cash_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Cash Ratio = Cash & Equivalents / Current Liabilities.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Cash Ratio values
        """
        cash = df.get('Cash and Equivalents', df.get('Efectivo y Equivalentes', df.get('Cash', pd.Series(index=df.index, dtype=float))))
        current_liabilities = df.get('Current Liabilities', df.get('Pasivo Corriente', pd.Series(index=df.index, dtype=float)))
        
        return self._safe_divide(cash, current_liabilities)
    
    def _compute_debt_to_equity(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Debt-to-Equity Ratio = Total Liabilities / Total Equity.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Debt-to-Equity Ratio values
        """
        total_liabilities = df.get('Total Liabilities', df.get('Pasivo Total', pd.Series(index=df.index, dtype=float)))
        total_equity = df.get('Total Equity', df.get('Patrimonio Neto', pd.Series(index=df.index, dtype=float)))
        
        # Handle negative equity (financial distress)
        # The ratio can be negative if equity < 0, which is a strong distress signal
        return self._safe_divide(total_liabilities, total_equity)
    
    def _compute_debt_to_assets(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Debt-to-Assets Ratio = Total Liabilities / Total Assets.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Debt-to-Assets Ratio values
        """
        total_liabilities = df.get('Total Liabilities', df.get('Pasivo Total', pd.Series(index=df.index, dtype=float)))
        total_assets = df.get('Total Assets', df.get('Activo Total', pd.Series(index=df.index, dtype=float)))
        
        # Should not exceed 1.0 (all assets financed by debt)
        ratio = self._safe_divide(total_liabilities, total_assets)
        ratio = ratio.clip(upper=1.0)  # Cap at 1.0
        
        return ratio
    
    def _compute_interest_coverage(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Interest Coverage = EBIT / Interest Expense.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Interest Coverage values
            
        Note:
            If Interest Expense is missing or zero, returns NaN.
            Negative EBIT results in negative coverage (distress signal).
        """
        ebit = df.get('EBIT', df.get('EBITDA', df.get('Resultado Operativo', pd.Series(index=df.index, dtype=float))))
        interest_expense = df.get('Interest Expense', df.get('Gastos Financieros', pd.Series(index=df.index, dtype=float)))
        
        # Interest coverage should be positive for healthy companies
        return self._safe_divide(ebit, interest_expense)
    
    def _compute_roa(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Return on Assets (ROA) = Net Income / Total Assets.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with ROA values (typically between -0.5 and 0.2)
        """
        net_income = df.get('Net Income', df.get('Resultado Neto', pd.Series(index=df.index, dtype=float)))
        total_assets = df.get('Total Assets', df.get('Activo Total', pd.Series(index=df.index, dtype=float)))
        
        return self._safe_divide(net_income, total_assets)
    
    def _compute_roe(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Return on Equity (ROE) = Net Income / Total Equity.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with ROE values (can be very high or negative for distressed companies)
            
        Note:
            Negative equity results in extreme negative ROE, which is a distress signal.
        """
        net_income = df.get('Net Income', df.get('Resultado Neto', pd.Series(index=df.index, dtype=float)))
        total_equity = df.get('Total Equity', df.get('Patrimonio Neto', pd.Series(index=df.index, dtype=float)))
        
        return self._safe_divide(net_income, total_equity)
    
    def _compute_operating_margin(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Operating Margin = Operating Income / Revenue.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Operating Margin values (typically between -0.5 and 0.3)
        """
        operating_income = df.get('Operating Income', df.get('Resultado Operativo', pd.Series(index=df.index, dtype=float)))
        revenue = df.get('Revenue', df.get('Ingresos', df.get('Ventas', pd.Series(index=df.index, dtype=float))))
        
        return self._safe_divide(operating_income, revenue)
    
    def _compute_asset_turnover(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Asset Turnover = Revenue / Total Assets.
        
        Args:
            df: Wide-format DataFrame
            
        Returns:
            Series with Asset Turnover values (typically between 0.1 and 2.0)
        """
        revenue = df.get('Revenue', df.get('Ingresos', df.get('Ventas', pd.Series(index=df.index, dtype=float))))
        total_assets = df.get('Total Assets', df.get('Activo Total', pd.Series(index=df.index, dtype=float)))
        
        return self._safe_divide(revenue, total_assets)
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle extreme ratio values using winsorization.
        
        Winsorization clips values beyond the specified percentiles to prevent
        extreme values from dominating the model.
        
        Args:
            df: DataFrame with ratio columns
            
        Returns:
            DataFrame with winsorized ratio columns
        """
        logger.info("Winsorizing extreme values...")
        
        ratio_columns = list(self.RATIO_DEFINITIONS.keys())
        
        for col in ratio_columns:
            if col in df.columns:
                # Winsorize at specified percentiles
                lower, upper = self.winsorize_limits
                lower_val = df[col].quantile(lower)
                upper_val = df[col].quantile(upper)
                
                df[col] = df[col].clip(lower=lower_val, upper=upper_val)
                
                logger.debug(f"Winsorized {col}: [{lower_val:.4f}, {upper_val:.4f}]")
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """
        Add temporal features (momentum, trend, volatility) for each ratio.
        
        Temporal features capture dynamics beyond static ratios:
            - Momentum: Quarter-over-quarter change
            - Trend: Rolling average over window quarters
            - Volatility: Standard deviation over trailing quarters
        
        Args:
            df: DataFrame with ratio columns (sorted by ticker and period)
            window: Number of quarters for rolling calculations (default: 4)
            
        Returns:
            DataFrame with temporal feature columns added
            
        Example:
            >>> df_temporal = engine.add_temporal_features(df_with_ratios, window=4)
            >>> print('current_ratio_momentum' in df_temporal.columns)
            True
        """
        logger.info(f"Adding temporal features with window={window} quarters...")
        
        ratio_columns = list(self.RATIO_DEFINITIONS.keys())
        
        # Ensure data is sorted by ticker and period
        df = df.sort_values(['ticker', 'period']).reset_index(drop=True)
        
        for col in ratio_columns:
            if col in df.columns:
                # Group by ticker for temporal calculations
                grouped = df.groupby('ticker')[col]
                
                # Momentum: Quarter-over-quarter change
                df[f'{col}_momentum'] = grouped.pct_change()
                
                # Trend: Rolling average
                df[f'{col}_trend'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Volatility: Rolling standard deviation
                df[f'{col}_volatility'] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        logger.info(f"Added {len(ratio_columns) * 3} temporal features")
        
        return df
    
    def add_sector_adjustments(
        self, 
        df: pd.DataFrame, 
        sector_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Add sector-relative adjustments to ratios.
        
        Sector-relative metrics improve signal quality by removing industry-wide effects.
        For each ratio, the sector median is subtracted from the company's value.
        
        Args:
            df: DataFrame with ratio columns and 'ticker' column
            sector_mapping: Dictionary mapping ticker to sector
                           {'GGAL': 'Financials', 'PAMP': 'Energy', ...}
            
        Returns:
            DataFrame with sector-relative ratio columns added
            
        Example:
            >>> sector_map = {'GGAL': 'Financials', 'PAMP': 'Energy'}
            >>> df_adjusted = engine.add_sector_adjustments(df_with_temporal, sector_map)
            >>> print('current_ratio_sector_relative' in df_adjusted.columns)
            True
        """
        logger.info("Adding sector-relative adjustments...")
        
        # Map ticker to sector
        df['sector'] = df['ticker'].map(sector_mapping)
        
        ratio_columns = list(self.RATIO_DEFINITIONS.keys())
        
        for col in ratio_columns:
            if col in df.columns:
                # Calculate sector median for each period
                sector_median = df.groupby(['sector', 'period'])[col].transform('median')
                
                # Sector-relative: company value - sector median
                df[f'{col}_sector_relative'] = df[col] - sector_median
        
        logger.info(f"Added {len(ratio_columns)} sector-relative features")
        
        return df
    
    def get_ratio_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics for all computed ratios.
        
        This method is useful for exploratory data analysis and identifying
        potential data quality issues.
        
        Args:
            df: DataFrame with ratio columns
            
        Returns:
            DataFrame with descriptive statistics:
                - count, mean, std, min, quartiles, max
                - missing values
                - outlier counts (beyond 1st/99th percentiles)
        """
        ratio_columns = list(self.RATIO_DEFINITIONS.keys())
        existing_cols = [col for col in ratio_columns if col in df.columns]
        
        if not existing_cols:
            logger.warning("No ratio columns found in DataFrame")
            return pd.DataFrame()
        
        stats = df[existing_cols].describe().T
        
        # Add missing value counts
        stats['missing'] = df[existing_cols].isna().sum()
        
        # Add outlier counts
        for col in existing_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            stats.loc[col, 'outliers'] = ((df[col] < lower) | (df[col] > upper)).sum()
        
        logger.info(f"Generated statistics for {len(existing_cols)} ratios")
        
        return stats
    
    def validate_ratios(self, df: pd.DataFrame) -> Dict:
        """
        Validate financial ratios for logical consistency.
        
        Validation rules:
            - Debt-to-Assets ratio should be <= 1.0
            - Ratios should fall within typical ranges (with some tolerance)
            - No ratio should have >50% missing values
        
        Args:
            df: DataFrame with ratio columns
            
        Returns:
            Dictionary with validation results:
                {'is_valid': bool, 'anomalies': [], 'missing_threshold_exceeded': []}
        """
        validation = {
            'is_valid': True,
            'anomalies': [],
            'missing_threshold_exceeded': []
        }
        
        ratio_columns = list(self.RATIO_DEFINITIONS.keys())
        
        for col in ratio_columns:
            if col not in df.columns:
                continue
                
            # Check for excessive missing values
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > 0.5:
                validation['missing_threshold_exceeded'].append({
                    'ratio': col,
                    'missing_pct': missing_pct
                })
                validation['is_valid'] = False
            
            # Check for values outside typical range
            if col in self.RATIO_DEFINITIONS:
                typical_min, typical_max = self.RATIO_DEFINITIONS[col]['typical_range']
                
                # Allow some tolerance (2x range)
                anomalies = df[(df[col] < typical_min * 2) | (df[col] > typical_max * 2)]
                
                if len(anomalies) > 0:
                    validation['anomalies'].append({
                        'ratio': col,
                        'count': len(anomalies),
                        'min_value': df[col].min(),
                        'max_value': df[col].max()
                    })
        
        if validation['anomalies'] or validation['missing_threshold_exceeded']:
            logger.warning(f"Validation issues found: {validation}")
        
        return validation


def main():
    """Command-line interface for ratio calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate financial ratios')
    parser.add_argument('--input', required=True, help='Input CSV file or directory')
    parser.add_argument('--output', default='data/processed/features.csv', help='Output CSV file')
    parser.add_argument('--sector-mapping', help='JSON file with sector mappings')
    parser.add_argument('--temporal-window', type=int, default=4, help='Temporal window in quarters')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = FinancialRatioEngine()
    
    # Load data
    df = engine.load_data(args.input)
    
    # Compute ratios
    df_ratios = engine.compute_all_ratios(df)
    
    # Add temporal features
    df_temporal = engine.add_temporal_features(df_ratios, window=args.temporal_window)
    
    # Add sector adjustments if mapping provided
    if args.sector_mapping:
        with open(args.sector_mapping, 'r', encoding='utf-8') as f:
            sector_mapping = json.load(f)
        df_final = engine.add_sector_adjustments(df_temporal, sector_mapping)
    else:
        df_final = df_temporal
    
    # Validate ratios
    validation = engine.validate_ratios(df_final)
    logger.info(f"Validation result: {validation}")
    
    # Save to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    logger.info(f"Features saved to {output_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())