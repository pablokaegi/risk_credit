"""
Yahoo Finance Data Fetcher for Argentine ADRs

This module fetches real financial data from Yahoo Finance for Argentine companies
with American Depositary Receipts (ADRs) trading on US markets.

Data includes:
    - Balance Sheet (quarterly/annual)
    - Income Statement (quarterly/annual)
    - Cash Flow Statement
    - Financial ratios (pre-calculated by Yahoo)
    - Market data (price, volume, etc.)

Benefits over CNV scraper:
    - No web scraping required (stable API)
    - Real data available immediately
    - Quarterly and annual data from 2010-present
    - Pre-formatted financial statements
    
Limitations:
    - Only covers ~15 Argentine companies with ADRs
    - Does not include companies trading only on BYMA
    
Author: Pablo Kaegi
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not installed. Install with: pip install yfinance")


logger = logging.getLogger(__name__)


# Argentine companies with ADRs on Yahoo Finance
ARGENTINE_ADRS = {
    'GGAL': {'name': 'Grupo Financiero Galicia', 'sector': 'Financials'},
    'PAMP': {'name': 'Pampa Energía', 'sector': 'Utilities'},
    'TGS': {'name': 'Transportadora de Gas del Sur', 'sector': 'Energy'},
    'BBAR': {'name': 'Banco BBVA Argentina', 'sector': 'Financials'},
    'LOMA': {'name': 'Loma Negra', 'sector': 'Basic Materials'},
    'CEPU': {'name': 'Central Puerto', 'sector': 'Utilities'},
    'TS': {'name': 'Tenaris', 'sector': 'Energy'},
    'SUPV': {'name': 'Grupo Supervielle', 'sector': 'Financials'},
    'BMA': {'name': 'Banco Macro', 'sector': 'Financials'},
    'IRS': {'name': 'IRSA Inversiones y Representaciones', 'sector': 'Real Estate'},
    'EDN': {'name': 'Edenor', 'sector': 'Utilities'},
    'CRESY': {'name': 'Cresud', 'sector': 'Real Estate'},
    'YPF': {'name': 'YPF', 'sector': 'Energy'},
}


class YFinanceDataFetcher:
    """
    Fetch financial data from Yahoo Finance for Argentine ADRs.
    
    This class provides a simple interface to download:
        - Balance Sheet (quarterly and annual)
        - Income Statement (quarterly and annual)
        - Cash Flow Statement
        - Pre-calculated financial ratios
        - Historical market data
    
    Attributes:
        tickers (list): List of ticker symbols to fetch
        frequency (str): 'quarterly' or 'annual'
        
    Example:
        >>> fetcher = YFinanceDataFetcher(tickers=['GGAL', 'PAMP', 'TGS'])
        >>> data = fetcher.fetch_all_data()
        >>> data['GGAL']['balance_sheet_quarterly'].head()
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        frequency: str = 'quarterly',
        output_dir: Union[str, Path] = 'data/raw/yfinance/'
    ):
        """
        Initialize the Yahoo Finance data fetcher.
        
        Args:
            tickers: List of ticker symbols. If None, uses all Argentine ADRs.
            frequency: 'quarterly' or 'annual' data frequency.
            output_dir: Directory to save downloaded data.
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )
        
        self.tickers = tickers if tickers else list(ARGENTINE_ADRS.keys())
        self.frequency = frequency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"YFinanceDataFetcher initialized with {len(self.tickers)} tickers")
    
    def fetch_all_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_to_csv: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all financial data for all tickers.
        
        Args:
            start_date: Start date for historical data (format: 'YYYY-MM-DD')
            end_date: End date for historical data (format: 'YYYY-MM-DD')
            save_to_csv: Whether to save data to CSV files
            
        Returns:
            Dictionary with structure: {ticker: {'balance_sheet': df, 'income_stmt': df, ...}}
            
        Example:
            >>> data = fetcher.fetch_all_data(start_date='2020-01-01')
            >>> ggal_balance = data['GGAL']['balance_sheet_quarterly']
        """
        all_data = {}
        
        for i, ticker in enumerate(self.tickers, 1):
            logger.info(f"Fetching data for {ticker} ({i}/{len(self.tickers)})")
            
            try:
                ticker_data = self.fetch_single_ticker(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                all_data[ticker] = ticker_data
                
                if save_to_csv:
                    self._save_ticker_data(ticker, ticker_data)
                    
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(all_data)} tickers")
        
        if save_to_csv:
            # Create combined dataset
            self._create_combined_dataset(all_data)
        
        return all_data
    
    def fetch_single_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'GGAL')
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dictionary with balance sheets, income statements, cash flows, and info
        """
        logger.info(f"Fetching data for {ticker}...")
        
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        data = {}
        
        # Fetch financial statements
        try:
            quarterly_balance_sheet = getattr(stock, 'quarterly_balance_sheet', stock.balance_sheet)
            quarterly_income_stmt = getattr(stock, 'quarterly_income_stmt', stock.financials)
            quarterly_cash_flow = getattr(
                stock,
                'quarterly_cash_flow',
                getattr(stock, 'quarterly_cashflow', stock.cashflow)
            )

            # Quarterly financials
            data['balance_sheet_quarterly'] = quarterly_balance_sheet
            data['income_stmt_quarterly'] = quarterly_income_stmt
            data['cash_flow_quarterly'] = quarterly_cash_flow
            
            # Annual financials
            data['balance_sheet_annual'] = stock.balance_sheet
            data['income_stmt_annual'] = getattr(stock, 'income_stmt', stock.financials)
            data['cash_flow_annual'] = getattr(stock, 'cash_flow', stock.cashflow)
            
            logger.debug(f"Retrieved financial statements for {ticker}")
            
        except Exception as e:
            logger.warning(f"Could not fetch financial statements for {ticker}: {e}")
        
        # Fetch historical market data
        try:
            if start_date and end_date:
                data['price_history'] = stock.history(start=start_date, end=end_date)
            else:
                # Default to last5 years
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                data['price_history'] = stock.history(start=start_date, end=end_date)
                
            logger.debug(f"Retrieved price history for {ticker}")
            
        except Exception as e:
            logger.warning(f"Could not fetch price history for {ticker}: {e}")
        
        # Fetch company info
        try:
            data['info'] = stock.info
            logger.debug(f"Retrieved company info for {ticker}")
        except Exception as e:
            logger.warning(f"Could not fetch company info for {ticker}: {e}")
        
        return data
    
    def _save_ticker_data(self, ticker: str, data: Dict[str, pd.DataFrame]) -> None:
        """
        Save ticker data to CSV files.
        
        Args:
            ticker: Stock ticker symbol
            data: Dictionary with DataFrames to save
        """
        ticker_dir = self.output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = ticker_dir / f"{name}.csv"
                df.to_csv(filepath)
                logger.debug(f"Saved {name} for {ticker} to {filepath}")
    
    def _create_combined_dataset(self, all_data: Dict) -> None:
        """
        Create a combined dataset with all tickers and periods.
        
        This creates a flat CSV file suitable for ML training with columns:
        ticker, period, all financial metrics...
        
        Args:
            all_data: Dictionary with all ticker data
        """
        logger.info("Creating combined dataset...")
        
        rows = []
        
        for ticker, ticker_data in all_data.items():
            # Use quarterly data for more granular analysis
            if 'balance_sheet_quarterly' not in ticker_data:
                continue
                
            bs = ticker_data['balance_sheet_quarterly']
            income = ticker_data.get('income_stmt_quarterly', pd.DataFrame())
            
            if bs.empty or income.empty:
                continue
            
            # Balance sheet columns (transposed - dates as columns)
            for date in bs.columns:
                try:
                    row = {'ticker': ticker, 'period': date.strftime('%Y-%m-%d')}
                    
                    # Add balance sheet items
                    # Map Yahoo Finance field names to our standard names
                    field_mappings = {
                        'Total Assets': 'total_assets',
                        'Total Current Assets': 'current_assets',
                        'Cash And Cash Equivalents': 'cash',
                        'Inventory': 'inventory',
                        'Total Liabilities Net Minority Interest': 'total_liabilities',
                        'Total Current Liabilities': 'current_liabilities',
                        'Total Debt': 'total_debt',
                        'Interest Expense': 'interest_expense',
                        'Total Equity Gross Minority Interest': 'total_equity',
                        'Retained Earnings': 'retained_earnings',
                        'Total Revenue': 'revenue',
                        'Operating Income': 'operating_income',
                        'Net Income': 'net_income',
                        'EBIT': 'ebit',
                        'EBITDA': 'ebitda',
                    }
                    
                    # Balance Sheet
                    for yf_field, std_field in field_mappings.items():
                        if yf_field in bs.index:
                            value = bs.loc[yf_field, date]
                            row[std_field] = value if not pd.isna(value) else None
                    
                    # Income Statement
                    for yf_field, std_field in field_mappings.items():
                        if yf_field in income.index:
                            value = income.loc[yf_field, date]
                            row[std_field] = value if not pd.isna(value) else None
                    
                    rows.append(row)
                    
                except Exception as e:
                    logger.debug(f"Error processing {ticker} for {date}: {e}")
                    continue
        
        if rows:
            combined_df = pd.DataFrame(rows)
            filepath = self.output_dir / 'combined_financials.csv'
            combined_df.to_csv(filepath, index=False)
            logger.info(f"Combined dataset saved to {filepath} ({len(combined_df)} rows)")
        else:
            logger.warning("No data to create combined dataset")
    
    def calculate_ratios_from_yahoo(
        self,
        ticker_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate financial ratios from Yahoo Finance data.
        
        This method extracts key financial metrics and computes the 10 ratios
        used for distress prediction.
        
        Args:
            ticker_data: Dictionary with balance sheets and income statements
            
        Returns:
            DataFrame with calculated ratios for each period
        """
        logger.info("Calculating financial ratios from Yahoo Finance data...")
        
        bs = ticker_data.get('balance_sheet_quarterly', pd.DataFrame())
        income = ticker_data.get('income_stmt_quarterly', pd.DataFrame())
        
        if bs.empty or income.empty:
            logger.warning("Insufficient data to calculate ratios")
            return pd.DataFrame()
        
        ratios_list = []
        
        # Process each period
        for date in bs.columns:
            try:
                # Extract values
                current_assets = self._safe_get(bs, 'Total Current Assets', date)
                cash = self._safe_get(bs, 'Cash And Cash Equivalents', date)
                inventory = self._safe_get(bs, 'Inventory', date, default=0)
                total_assets = self._safe_get(bs, 'Total Assets', date)
                
                current_liabilities = self._safe_get(bs, 'Total Current Liabilities', date)
                total_liabilities = self._safe_get(bs, 'Total Liabilities Net Minority Interest', date)
                total_equity = self._safe_get(bs, 'Total Equity Gross Minority Interest', date)
                
                revenue = self._safe_get(income, 'Total Revenue', date)
                operating_income = self._safe_get(income, 'Operating Income', date)
                net_income = self._safe_get(income, 'Net Income', date)
                ebit = self._safe_get(income, 'EBIT', date, default=operating_income)
                ebitda = self._safe_get(income, 'EBITDA', date)
                interest_expense = abs(self._safe_get(income, 'Interest Expense', date, default=0))
                
                # Calculate ratios
                ratios = {
                    'period': date.strftime('%Y-%m-%d'),
                    
                    # Liquidity ratios
                    'current_ratio': self._safe_divide(current_assets, current_liabilities),
                    'quick_ratio': self._safe_divide(current_assets - inventory, current_liabilities),
                    'cash_ratio': self._safe_divide(cash, current_liabilities),
                    
                    # Leverage ratios
                    'debt_to_equity': self._safe_divide(total_liabilities, total_equity),
                    'debt_to_assets': self._safe_divide(total_liabilities, total_assets),
                    'interest_coverage': self._safe_divide(ebit, interest_expense),
                    
                    # Profitability ratios
                    'roa': self._safe_divide(net_income, total_assets),
                    'roe': self._safe_divide(net_income, total_equity),
                    'operating_margin': self._safe_divide(operating_income, revenue),
                    
                    # Efficiency ratios
                    'asset_turnover': self._safe_divide(revenue, total_assets),
                    
                    # Raw values for reference
                    'total_assets': total_assets,
                    'total_equity': total_equity,
                    'total_liabilities': total_liabilities,
                    'revenue': revenue,
                    'net_income': net_income,
                }
                
                ratios_list.append(ratios)
                
            except Exception as e:
                logger.debug(f"Error calculating ratios for {date}: {e}")
                continue
        
        df = pd.DataFrame(ratios_list)
        logger.info(f"Calculated ratios for {len(df)} periods")
        
        return df
    
    def _safe_get(
        self,
        df: pd.DataFrame,
        field: str,
        date,
        default: float = None
    ) -> float:
        """
        Safely extract value from DataFrame.
        
        Args:
            df: DataFrame to extract from
            field: Field name to extract
            date: Date column
            default: Default value if field not found
            
        Returns:
            Float value or default
        """
        try:
            if field in df.index:
                value = df.loc[field, date]
                return float(value) if not pd.isna(value) else default
            return default
        except:
            return default
    
    def _safe_divide(
        self,
        numerator: float,
        denominator: float,
        default: float = None
    ) -> float:
        """
        Safely divide two numbers, handling division by zero.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division invalid
            
        Returns:
            Division result or default
        """
        if numerator is None or denominator is None:
            return default
        if denominator == 0:
            return default
        return numerator / denominator
    
    def create_target_variable(
        self,
        combined_df: pd.DataFrame,
        method: str = 'equity_negative',
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Create target variable for distress prediction.
        
        Methods available:
            - 'equity_negative': Target=1 if Total Equity < 0
            - 'debt_ratio': Target=1 if Debt/Equity > threshold (default: 5)
            - 'combined': Target=1 if ANY distress criteria met
        
        Args:
            combined_df: DataFrame with financial data
            method: Method to define distress
            threshold: Threshold for debt_ratio method
            
        Returns:
            DataFrame with 'target' column added
        """
        logger.info(f"Creating target variable using method: {method}")
        
        df = combined_df.copy()
        
        if method == 'equity_negative':
            # Simple: negative equity
            df['target'] = (df['total_equity'] < 0).astype(int)
            df['distress_reason'] = 'negative_equity'
            
        elif method == 'debt_ratio':
            # High leverage
            if threshold is None:
                threshold =5.0
            df['target'] = (df['debt_to_equity'] > threshold).astype(int)
            df['distress_reason'] = 'high_leverage'
            
        elif method == 'combined':
            # Multiple criteria
            distress_conditions = (
                (df['total_equity'] <0) |  # Negative equity
                (df['debt_to_equity'] > 5) |  # High leverage
                (df['interest_coverage'] < 1.5) |  # Low coverage
                (df['roa'] < -0.1) |  # Losses
                (df['current_ratio'] <0.5)  # Liquidity crisis
            )
            df['target'] = distress_conditions.astype(int)
            df['distress_reason'] = 'multiple_factors'
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Log distribution
        target_dist = df['target'].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")
        logger.info(f"Distress rate: {(df['target'] == 1).mean():.2%}")
        
        return df
    
    def get_available_tickers(self) -> List[Dict]:
        """
        Get list of available Argentine ADRs.
        
        Returns:
            List of dictionaries with ticker info
        """
        return [
            {'ticker': ticker, **info}
            for ticker, info in ARGENTINE_ADRS.items()
        ]


def download_and_prepare_dataset(
    tickers: Optional[List[str]] = None,
    output_path: Union[str, Path] = 'data/processed/dataset.csv',
    start_date: str = '2019-01-01',
    target_method: str = 'combined'
) -> pd.DataFrame:
    """
    Convenience function to download, process, and save complete dataset.
    
    This is the main entry point for getting a ready-to-use dataset.
    
    Args:
        tickers: List of tickers (default: all Argentine ADRs)
        output_path: Where to save the processed dataset
        start_date: Start date for historical data
        target_method: Method for target variable creation
        
    Returns:
        DataFrame ready for ML training
        
    Example:
        >>> df = download_and_prepare_dataset(tickers=['GGAL', 'PAMP'], start_date='2020-01-01')
        >>> print(df.shape)
        (120, 15)  # 2 tickers × 60 quarters
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING AND PREPARING DATASET FROM YAHOO FINANCE")
    logger.info("=" * 80)
    
    # Initialize fetcher
    fetcher = YFinanceDataFetcher(tickers=tickers, frequency='quarterly')
    
    # Download data
    all_data = fetcher.fetch_all_data(
        start_date=start_date,
        save_to_csv=True
    )
    
    # Calculate ratios for each ticker
    all_ratios = []
    
    for ticker, ticker_data in all_data.items():
        logger.info(f"Calculating ratios for {ticker}...")
        
        if 'balance_sheet_quarterly' in ticker_data:
            ratios_df = fetcher.calculate_ratios_from_yahoo(ticker_data)
            
            if not ratios_df.empty:
                ratios_df['ticker'] = ticker
                all_ratios.append(ratios_df)
    
    # Combine all ratios
    if all_ratios:
        combined_df = pd.concat(all_ratios, ignore_index=True)
        
        # Create target variable
        final_df = fetcher.create_target_variable(
            combined_df,
            method=target_method
        )
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Shape: {final_df.shape}")
        logger.info(f"Columns: {list(final_df.columns)}")
        
        return final_df
    else:
        logger.error("No data to create dataset")
        return pd.DataFrame()


def main():
    """Command-line interface for Yahoo Finance data fetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download financial data from Yahoo Finance for Argentine ADRs'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='List of tickers (default: all Argentine ADRs)'
    )
    parser.add_argument(
        '--output',
        default='data/processed/dataset.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--start-date',
        default='2019-01-01',
        help='Start date for historical data (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--target-method',
        default='combined',
        choices=['equity_negative', 'debt_ratio', 'combined'],
        help='Method for target variable creation'
    )
    
    args = parser.parse_args()
    
    # Run
    df = download_and_prepare_dataset(
        tickers=args.tickers,
        output_path=args.output,
        start_date=args.start_date,
        target_method=args.target_method
    )
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())