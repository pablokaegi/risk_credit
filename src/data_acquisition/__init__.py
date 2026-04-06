"""
Data Acquisition Module

This module handles automated extraction of financial data from Argentine
regulatory sources:
    - BCRA APIs (PRIMARY): Official Central Bank data (Central de Deudores, Cheques Denunciados)
    - CNV Web Scraping (BACKUP): Financial statements from CNV Autoconvocatoria
    - Yahoo Finance (FALLBACK): ADR data for Argentine companies
"""

from src.data_acquisition.bcra_api import BCRAAPIClient, create_training_dataset

try:
    from src.data_acquisition.yfinance_fetcher import (
        YFinanceDataFetcher,
        download_and_prepare_dataset,
    )
except ImportError:  # pragma: no cover - optional dependency in partial installs
    YFinanceDataFetcher = None
    download_and_prepare_dataset = None

__all__ = [
    "BCRAAPIClient",
    "create_training_dataset",
    "YFinanceDataFetcher",
    "download_and_prepare_dataset",
]