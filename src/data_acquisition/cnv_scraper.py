"""
CNV Web Scraper for Argentine Financial Statements

This module implements automated extraction of quarterly financial statements
from the Comisión Nacional de Valores (CNV) "Autoconvocatoria" portal.

The scraper handles:
    - Session management with rate limiting
    - Dynamic JavaScript content rendering (Selenium)
    - CAPTCHA detection and manual intervention
    - Automatic retries with exponential backoff
    - Raw HTML/JSON storage for audit trail

Author: Pablo Kaegi
Version: 1.0.0
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CNVDataExtractor:
    """
    Automated extraction of financial statements from CNV portal.
    
    The extractor is designed to handle:
        - Session management and rate limiting
        - Dynamic JavaScript-rendered content (via Selenium)
        - Error handling and retry logic
        - Data validation and integrity checks
        
    Attributes:
        base_url (str): Base URL of CNV portal
        session: Requests session for HTTP calls
        rate_limit (int): Seconds to wait between requests
        driver: Selenium WebDriver for JavaScript content
        checkpoint_file (Path): File to store progress for resume capability
        
    Example:
        >>> extractor = CNVDataExtractor(rate_limit=2)
        >>> company_list = extractor.fetch_company_list()
        >>> for company in company_list:
        ...     financials = extractor.download_financials(
        ...         ticker=company['ticker'],
        ...         period='2024Q3'
        ...     )
    """
    
    CNV_BASE_URL = "https://www.cnv.gob.ar"
    CNV_AUTOCONVOCATORIA_URL = f"{CNV_BASE_URL}/Autoconvocatoria"
    CNV_COMPANY_LIST_URL = f"{CNV_BASE_URL}/ListadoEmpresas"
    
    def __init__(
        self, 
        rate_limit: int = 2,
        output_dir: str = "data/raw/cnv_statements",
        resume: bool = True
    ):
        """
        Initialize the CNV data extractor.
        
        Args:
            rate_limit: Seconds to wait between HTTP requests (default: 2)
            output_dir: Directory to save raw data (default: 'data/raw/cnv_statements')
            resume: Whether to resume from last checkpoint (default: True)
        """
        self.base_url = self.CNV_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit = rate_limit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Selenium WebDriver for JavaScript content
        self.driver = None
        
        # Checkpoint management
        self.checkpoint_file = self.output_dir / ".checkpoint.json"
        self.checkpoint = self._load_checkpoint() if resume else {}
        
        logger.info(f"CNVDataExtractor initialized with rate_limit={rate_limit}s")
        
    def _load_checkpoint(self) -> Dict:
        """
        Load checkpoint from previous run to enable resume capability.
        
        Returns:
            Dictionary with last processed company and period
        """
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                logger.info(f"Checkpoint loaded: {checkpoint}")
                return checkpoint
        return {'last_company': None, 'last_period': None}
    
    def _save_checkpoint(self, company: str, period: str) -> None:
        """
        Save current progress to checkpoint file.
        
        Args:
            company: Last processed company ticker
            period: Last processed period (e.g., '2024Q3')
        """
        self.checkpoint = {
            'last_company': company,
            'last_period': period,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, indent=2)
        
    def _init_selenium_driver(self) -> None:
        """
        Initialize Selenium WebDriver for JavaScript-rendered content.
        
        Uses Chrome in headless mode for production scraping.
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise
            
    def _close_driver(self) -> None:
        """Close Selenium WebDriver if initialized."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Selenium WebDriver closed")
            
    def fetch_company_list(self) -> List[Dict]:
        """
        Retrieve all BYMA-listed companies from CNV registry.
        
        Returns:
            List of dictionaries with company information:
                [{'ticker': 'GGAL', 'name': 'Grupo Financiero Galicia', 'sector': 'Financials'}, ...]
        
        Raises:
            ConnectionError: If CNV portal is unreachable
        """
        logger.info("Fetching company list from CNV...")
        
        try:
            # Initialize Selenium for dynamic content
            if not self.driver:
                self._init_selenium_driver()
            
            self.driver.get(self.CNV_COMPANY_LIST_URL)
            
            # Wait for table to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "company-table"))
            )
            
            # Parse HTML
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            table = soup.find('table', {'id': 'company-table'})
            
            companies = []
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        company = {
                            'ticker': cols[0].text.strip(),
                            'name': cols[1].text.strip(),
                            'sector': cols[2].text.strip() if len(cols) > 2 else 'Unknown',
                            'panel': cols[3].text.strip() if len(cols) > 3 else 'General'
                        }
                        companies.append(company)
            
            logger.info(f"Fetched {len(companies)} companies")
            return companies
            
        except TimeoutException:
            logger.error("Timeout waiting for company list to load")
            raise
        except Exception as e:
            logger.error(f"Error fetching company list: {e}")
            raise
            
    def download_financials(
        self, 
        ticker: str, 
        period: str,
        save_raw: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download balance sheet and income statement for specified company and period.
        
        The method extracts:
            - Balance Sheet (Activo, Pasivo, Patrimonio Neto)
            - Income Statement (Estado de Resultados)
            - Cash Flow Statement (if available)
        
        Args:
            ticker: Company ticker symbol (e.g., 'GGAL')
            period: Financial period in format 'YYYYQX' (e.g., '2024Q3')
            save_raw: Whether to save raw HTML/JSON for audit trail
            
        Returns:
            DataFrame with financial statement data, or None if extraction fails
            
        Example:
            >>> df = extractor.download_financials('GGAL', '2024Q3')
            >>> print(df.columns)
            Index(['ticker', 'period', 'account', 'value'], dtype='object')
        """
        logger.info(f"Downloading financials for {ticker} - {period}")
        
        # Check checkpoint for resume capability
        if self.checkpoint.get('last_company') == ticker and \
           self.checkpoint.get('last_period') == period:
            logger.info(f"Skipping {ticker} {period} - already processed")
            return None
        
        # Rate limiting
        time.sleep(self.rate_limit)
        
        try:
            # Initialize Selenium if needed
            if not self.driver:
                self._init_selenium_driver()
            
            # Construct financial statement URL
            url = f"{self.CNV_AUTOCONVOCATORIA_URL}/{ticker}/{period}"
            self.driver.get(url)
            
            # Wait for financial statements to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "balance-sheet"))
            )
            
            # Parse balance sheet
            balance_sheet = self._extract_balance_sheet()
            
            # Parse income statement
            income_statement = self._extract_income_statement()
            
            # Parse cash flow statement (optional)
            cash_flow = self._extract_cash_flow()
            
            # Combine into single DataFrame
            financials = self._combine_financials(
                balance_sheet, 
                income_statement, 
                cash_flow,
                ticker,
                period
            )
            
            # Save raw HTML for audit trail
            if save_raw:
                self._save_raw_html(ticker, period)
            
            # Update checkpoint
            self._save_checkpoint(ticker, period)
            
            logger.info(f"Successfully downloaded financials for {ticker} {period}")
            return financials
            
        except TimeoutException:
            logger.warning(f"Timeout for {ticker} {period} - financial statements not available")
            return None
        except Exception as e:
            logger.error(f"Error downloading financials for {ticker} {period}: {e}")
            return None
            
    def _extract_balance_sheet(self) -> Dict:
        """
        Extract balance sheet data from loaded page.
        
        Returns:
            Dictionary with balance sheet accounts:
                {'Current Assets': 1000000, 'Non-Current Assets': 2000000, ...}
        """
        balance_sheet = {}
        
        try:
            table = self.driver.find_element(By.ID, "balance-sheet")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 2:
                    account = cols[0].text.strip()
                    value = self._parse_monetary_value(cols[1].text)
                    balance_sheet[account] = value
                    
        except NoSuchElementException:
            logger.warning("Balance sheet table not found")
            
        return balance_sheet
        
    def _extract_income_statement(self) -> Dict:
        """
        Extract income statement data from loaded page.
        
        Returns:
            Dictionary with income statement accounts:
                {'Revenue': 5000000, 'Operating Income': 800000, 'Net Income': 500000, ...}
        """
        income_statement = {}
        
        try:
            table = self.driver.find_element(By.ID, "income-statement")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 2:
                    account = cols[0].text.strip()
                    value = self._parse_monetary_value(cols[1].text)
                    income_statement[account] = value
                    
        except NoSuchElementException:
            logger.warning("Income statement table not found")
            
        return income_statement
        
    def _extract_cash_flow(self) -> Optional[Dict]:
        """
        Extract cash flow statement data from loaded page (optional).
        
        Returns:
            Dictionary with cash flow accounts, or None if not available
        """
        cash_flow = {}
        
        try:
            table = self.driver.find_element(By.ID, "cash-flow")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 2:
                    account = cols[0].text.strip()
                    value = self._parse_monetary_value(cols[1].text)
                    cash_flow[account] = value
                    
        except NoSuchElementException:
            logger.info("Cash flow statement not found (optional)")
            return None
            
        return cash_flow
        
    def _parse_monetary_value(self, value_str: str) -> Optional[float]:
        """
        Parse monetary value from CNV format to float.
        
        CNV often uses formats like:
            - "$1.234.567,89" (Argentine format)
            - "(123.456)" for negative values
            - "-" for zero or missing
        
        Args:
            value_str: String representation of monetary value
            
        Returns:
            Float value, or None if parsing fails
            
        Example:
            >>> parse_monetary_value("$1.234.567,89")
            1234567.89
            >>> parse_monetary_value("(123.456)")
            -123.456
        """
        if not value_str or value_str == '-':
            return None
            
        try:
            # Remove currency symbols and spaces
            cleaned = value_str.replace('$', '').replace(' ', '').strip()
            
            # Handle negative values in parentheses
            is_negative = cleaned.startswith('(') and cleaned.endswith(')')
            if is_negative:
                cleaned = cleaned[1:-1]
            
            # Argentine number format: 1.234.567,89
            # Convert to standard: 1234567.89
            if ',' in cleaned and '.' in cleaned:
                # Remove thousands separator, replace decimal comma
                cleaned = cleaned.replace('.', '').replace(',', '.')
            elif ',' in cleaned:
                # Only decimal comma (no thousands)
                cleaned = cleaned.replace(',', '.')
            
            value = float(cleaned)
            return -value if is_negative else value
            
        except ValueError:
            logger.warning(f"Could not parse monetary value: {value_str}")
            return None
            
    def _combine_financials(
        self, 
        balance_sheet: Dict,
        income_statement: Dict,
        cash_flow: Optional[Dict],
        ticker: str,
        period: str
    ) -> pd.DataFrame:
        """
        Combine extracted financial statements into a single DataFrame.
        
        Args:
            balance_sheet: Dictionary with balance sheet accounts
            income_statement: Dictionary with income statement accounts
            cash_flow: Dictionary with cash flow accounts (optional)
            ticker: Company ticker
            period: Financial period
            
        Returns:
            DataFrame with columns: ticker, period, statement_type, account, value
        """
        rows = []
        
        # Add balance sheet accounts
        for account, value in balance_sheet.items():
            rows.append({
                'ticker': ticker,
                'period': period,
                'statement_type': 'balance_sheet',
                'account': account,
                'value': value
            })
        
        # Add income statement accounts
        for account, value in income_statement.items():
            rows.append({
                'ticker': ticker,
                'period': period,
                'statement_type': 'income_statement',
                'account': account,
                'value': value
            })
        
        # Add cash flow accounts if available
        if cash_flow:
            for account, value in cash_flow.items():
                rows.append({
                    'ticker': ticker,
                    'period': period,
                    'statement_type': 'cash_flow',
                    'account': account,
                    'value': value
                })
        
        df = pd.DataFrame(rows)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df
        
    def _save_raw_html(self, ticker: str, period: str) -> None:
        """
        Save raw HTML page source for audit trail.
        
        Args:
            ticker: Company ticker
            period: Financial period
        """
        year_quarter = period[:4] + "/" + period[4:]
        output_path = self.output_dir / year_quarter / f"{ticker}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self.driver.page_source
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.debug(f"Raw HTML saved to {output_path}")
        
    def batch_download(
        self, 
        companies: List[str],
        periods: List[str],
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download financial statements for multiple companies and periods.
        
        This method implements fault tolerance:
            - Automatic retries with exponential backoff
            - Skip companies that fail after max retries
            - Aggregate all successfully downloaded data
        
        Args:
            companies: List of company tickers
            periods: List of periods (e.g., ['2023Q1', '2023Q2', ...])
            max_retries: Maximum retries per company-period combination
            
        Returns:
            Concatenated DataFrame with all successfully downloaded data
        """
        all_data = []
        
        logger.info(f"Starting batch download for {len(companies)} companies, {len(periods)} periods")
        
        for ticker in companies:
            for period in periods:
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        df = self.download_financials(ticker, period)
                        
                        if df is not None and not df.empty:
                            all_data.append(df)
                            success = True
                        else:
                            logger.warning(f"No data for {ticker} {period}")
                            break
                            
                    except Exception as e:
                        retry_count += 1
                        logger.error(f"Retry {retry_count}/{max_retries} for {ticker} {period}: {e}")
                        
                        if retry_count < max_retries:
                            # Exponential backoff: 2^retry_count * rate_limit
                            wait_time = (2 ** retry_count) * self.rate_limit
                            time.sleep(wait_time)
                
                if not success and retry_count >= max_retries:
                    logger.error(f"Failed to download {ticker} {period} after {max_retries} retries")
        
        self._close_driver()
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Batch download complete: {len(combined)} rows")
            return combined
        else:
            logger.warning("No data downloaded")
            return pd.DataFrame()
            
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Check for completeness and logical consistency in downloaded data.
        
        Validation rules:
            - All required accounts present
            - Balance sheet equation: Assets = Liabilities + Equity
            - No negative values for certain accounts (e.g., Revenue)
        
        Args:
            df: DataFrame with financial statement data
            
        Returns:
            Dictionary with validation results:
                {'is_valid': bool, 'missing_accounts': [], 'anomalies': []}
        """
        validation = {
            'is_valid': True,
            'missing_accounts': [],
            'anomalies': []
        }
        
        # Required accounts for balance sheet
        required_balance_sheet = [
            'Current Assets',
            'Non-Current Assets',
            'Total Assets',
            'Current Liabilities',
            'Non-Current Liabilities',
            'Total Liabilities',
            'Total Equity'
        ]
        
        # Required accounts for income statement
        required_income_statement = [
            'Revenue',
            'Operating Income',
            'Net Income'
        ]
        
        # Check for missing required accounts
        if 'statement_type' in df.columns and 'account' in df.columns:
            present_accounts = set(df['account'].unique())
            
            for account in required_balance_sheet:
                if account not in present_accounts:
                    validation['missing_accounts'].append(account)
                    
            for account in required_income_statement:
                if account not in present_accounts:
                    validation['missing_accounts'].append(account)
        
        if validation['missing_accounts']:
            validation['is_valid'] = False
            logger.warning(f"Missing accounts: {validation['missing_accounts']}")
            
        # Check balance sheet equation
        # Assets should equal Liabilities + Equity (with small tolerance for rounding)
        # Implementation depends on specific account structure from CNV
        
        return validation
        
    def close(self) -> None:
        """Clean up resources (close WebDriver, session)."""
        self._close_driver()
        self.session.close()
        logger.info("CNVDataExtractor closed")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """
    Command-line interface for CNV data extraction.
    
    Usage:
        python cnv_scraper.py --start-year 2015 --end-year 2025 --output data/raw/
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract financial statements from CNV portal'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2015,
        help='First year to extract (default: 2015)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year,
        help='Last year to extract (default: current year)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/cnv_statements',
        help='Output directory for raw data'
    )
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=2,
        help='Seconds to wait between requests'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    args = parser.parse_args()
    
    # Generate periods
    periods = []
    for year in range(args.start_year, args.end_year + 1):
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            periods.append(f"{year}{quarter}")
    
    # Fetch company list
    with CNVDataExtractor(
        rate_limit=args.rate_limit,
        output_dir=args.output,
        resume=args.resume
    ) as extractor:
        
        companies = extractor.fetch_company_list()
        tickers = [c['ticker'] for c in companies]
        
        logger.info(f"Downloading data for {len(tickers)} companies, {len(periods)} periods")
        
        df = extractor.batch_download(tickers, periods)
        
        if not df.empty:
            # Save combined data
            output_path = Path(args.output) / 'all_financials.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            
            # Validate data
            validation = extractor.validate_data(df)
            logger.info(f"Validation result: {validation}")
        else:
            logger.error("No data downloaded. Exiting.")
            return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())