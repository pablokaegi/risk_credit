"""
BCRA API Client for Argentine Financial Data

This module provides a professional-grade interface to the Central Bank of Argentina's
official APIs, specifically designed for credit risk and financial distress analysis.

APIs Available:
    - Central de Deudores: Debt information for all Argentine companies
    - Cheques Denunciados: Checks rejected due to insufficient funds (distress signal)
    - Estadísticas Cambiarias: Exchange rate statistics
    - Estadísticas Monetarias: Monetary and banking statistics
    - Régimen de Transparencia: Transparency regime data

Official Documentation: https://api.bcra.gob.ar
Contact: api@bcra.gob.ar

This is the PRIMARY data source for Argentine credit risk analysis, superior to:
    - yFinance: Only covers ~15 ADRs, not local market
    - CNV scraping: Unstable, requires web automation
    - Manual collection: Slow, error-prone

Author: Pablo Kaegi
Version: 2.0.0 (Production-ready)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import time

import requests
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class BCRAAPIClient:
    """
    Professional API client for Banco Central de la República Argentina (BCRA).
    
    This client provides access to official Argentine financial data including:
        - Central de Deudores: Complete debt registry for all companies
        - Cheques Denunciados: Rejected checks (strong distress indicator)
        - Estadísticas Cambiarias: Exchange rates and currency reserves
        - Estadísticas Monetarias: Monetary aggregates and interest rates
        - Régimen de Transparencia: Regulatory transparency data
    
    Features:
        - Rate limiting protection
        - Automatic retry with exponential backoff
        - Response caching
        - Error handling and logging
        - DataFrame conversion for all endpoints
    
    Example:
        >>> client = BCRAAPIClient()
        >>> deudores = client.get_central_deudores(cuit='30-12345678-9')
        >>> cheque = client.get_cheques_denunciados(codigo_entidad=11, numero_cheque=20377516)
        >>> dollar_rate = client.get_dollar_rate()
    """
    
    BASE_URL = "https://api.bcra.gob.ar"
    
    ENDPOINTS = {
        # Central de Deudores - Debt information
        'central_deudores': '/api/v1/central/deudores',
        'central_deudores_entidades': '/api/v1/central/deudores/entidades',
        
        # Cheques Denunciados - Rejected checks
        'cheques_entidades': '/cheques/v1.0/entidades',
        'cheques_denunciados': '/cheques/v1.0/denunciados/{codigo_entidad}/{numero_cheque}',
        
        # Estadísticas Cambiarias - Exchange rates
        'estadisticas_cambiarias': '/api/v1/estadisticascambiarias',
        'dollar_oficial': '/api/v1/estadisticascambiarias/dolaroficial',
        'dollar_mayorista': '/api/v1/estadisticascambiarias/dolarmayorista',
        'reservas': '/api/v1/estadisticascambiarias/reservas',
        
        # Estadísticas Monetarias - Monetary statistics
        'estadisticas_monetarias': '/api/v1/estadisticasmonetarias',
        'tasas_interes': '/api/v1/estadisticasmonetarias/tasas',
        'agregados_monetarios': '/api/v1/estadisticasmonetarias/agregados',
        
        # Régimen de Transparencia - Transparency regime
        'transparencia': '/api/v1/transparencia',
        'transparencia_entidades': '/api/v1/transparencia/entidades/{entidad}',
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        retry_attempts: int = 3,
        cache_enabled: bool = True,
        cache_dir: Union[str, Path] = 'data/cache/bcra/'
    ):
        """
        Initialize BCRA API client.
        
        Args:
            api_key: Optional API key (some endpoints may require authentication)
            rate_limit_delay: Seconds to wait between requests (default: 1.0)
            retry_attempts: Number of retry attempts on failure (default: 3)
            cache_enabled: Whether to cache responses locally (default: True)
            cache_dir: Directory for response cache
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.retry_attempts = retry_attempts
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Argentine-Distress-Model/2.0'
        })
        
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        
        self.last_request_time = None
        
        logger.info("BCRAAPIClient initialized successfully")
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Make HTTP request to BCRA API with rate limiting and retry logic.
        
        Args:
            endpoint: API endpoint (from ENDPOINTS)
            params: Query parameters
            use_cache: Whether to use cached response if available
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If all retry attempts fail
        """
        url = f"{self.BASE_URL}{endpoint}"
        cache_file = self.cache_dir / f"{endpoint.replace('/', '_')}.json"
        
        # Check cache
        if self.cache_enabled and use_cache and cache_file.exists():
            logger.debug(f"Using cached response for {endpoint}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                import json
                return json.load(f)
        
        # Rate limiting
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        
        # Retry logic
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.debug(f"Request attempt {attempt}/{self.retry_attempts}: {url}")
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                self.last_request_time = datetime.now()
                
                data = response.json()
                
                # Cache response
                if self.cache_enabled:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(data, f, indent=2)
                
                return data

            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None

                if status_code is not None and 400 <= status_code < 500 and status_code not in {408, 429}:
                    logger.error(
                        f"Permanent client error for {url}: {e}. "
                        "Verify the published BCRA endpoint before retrying."
                    )
                    raise
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt}): {e}")
                
                if attempt == self.retry_attempts:
                    logger.error(f"All retry attempts failed for {url}")
                    raise
                
                # Exponential backoff
                wait_time = (2 ** attempt) * self.rate_limit_delay
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    def get_central_deudores(
        self,
        cuit: Optional[str] = None,
        entidad: Optional[str] = None,
        periodo: Optional[str] = None,
        as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get debt information from Central de Deudores.
        
        The Central de Deudores contains information about ALL debt held by
        individuals and companies in the Argentine financial system. This is
        the PRIMARY source for credit risk analysis.
        
        Args:
            cuit: Specific CUIT/CUIL to query (default: all)
            entidad: Financial entity code (default: all)
            periodo: Period in format 'YYYYMM' (default: latest)
            as_dataframe: Return as DataFrame (default: True)
            
        Returns:
            DataFrame with columns:
            - cuit: Tax ID (identifies the company)
            - entidad: Financial entity code
            - situacion: Debt situation (1-5, where 5 is worst)
            - deuda: Total debt amount
            - periodo: Reporting period
            
        Example:
            >>> client = BCRAAPIClient()
            >>> df = client.get_central_deudores(cuit='30-12345678-9')
            >>> print(df['deuda'].sum())  # Total debt for this company
        """
        params = {}
        
        if cuit:
            params['cuit'] = cuit
        if entidad:
            params['entidad'] = entidad
        if periodo:
            params['periodo'] = periodo
        
        data = self._make_request(self.ENDPOINTS['central_deudores'], params=params)
        
        if as_dataframe:
            return self._parse_central_deudores(data)
        
        return data
    
    def get_cheques_denunciados(
        self,
        codigo_entidad: int,
        numero_cheque: int,
        as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Query whether a specific check is reported as denounced.
        
        According to the official BCRA Cheques API manual, this endpoint requires
        both the financial entity code and the check number. The API does not
        expose bulk search by CUIT.
        
        Args:
            codigo_entidad: Entity code from get_cheques_entidades()
            numero_cheque: Check number to query
            as_dataframe: Return as DataFrame (default: True)
            
        Returns:
            DataFrame with columns:
            - numero_cheque: Check number
            - denunciado: Whether the check is denounced
            - fecha_procesamiento: Processing date
            - denominacion_entidad: Entity name
            - cantidad_detalles: Number of detail rows
            - causales: Semicolon-separated causes
            
        Example:
            >>> client = BCRAAPIClient()
            >>> cheque = client.get_cheques_denunciados(codigo_entidad=11, numero_cheque=20377516)
            >>> print(cheque[['numero_cheque', 'denunciado']])
        """
        endpoint = self.ENDPOINTS['cheques_denunciados'].format(
            codigo_entidad=int(codigo_entidad),
            numero_cheque=int(numero_cheque)
        )
        data = self._make_request(endpoint)
        
        if as_dataframe:
            return self._parse_cheques_denunciados(data)
        
        return data

    def get_cheques_entidades(self, as_dataframe: bool = True) -> Union[Dict, pd.DataFrame]:
        """Get the list of financial entities available in the Cheques API."""
        data = self._make_request(self.ENDPOINTS['cheques_entidades'])

        if as_dataframe:
            return self._parse_cheques_entidades(data)

        return data
    
    def get_dollar_rate(
        self,
        fecha_desde: Optional[str] = None,
        fecha_hasta: Optional[str] = None,
        tipo: str = 'oficial',
        as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get historical dollar exchange rates.
        
        Exchange rate is critical for Argentine companies with:
            - Dollar-denominated debt
            - Import/export operations
            - Inflation-adjusted accounting
        
        Args:
            fecha_desde: Start date 'YYYY-MM-DD' (default: last 90 days)
            fecha_hasta: End date 'YYYY-MM-DD' (default: today)
            tipo: 'oficial', 'mayorista', or 'blue' (default: 'oficial')
            as_dataframe: Return as DataFrame (default: True)
            
        Returns:
            DataFrame with columns:
            - fecha: Date
            - compra: Buy rate
            - venta: Sell rate
            - promedio: Average rate
            
        Example:
            >>> client = BCRAAPIClient()
            >>> dollar = client.get_dollar_rate(tipo='oficial')
            >>> print(dollar.tail())  # Last 5 days
        """
        endpoint_key = f'dollar_{tipo}' if tipo != 'blue' else 'dollar_oficial'
        
        if endpoint_key not in self.ENDPOINTS:
            logger.warning(f"Dollar type '{tipo}' not available, using 'oficial'")
            endpoint_key = 'dollar_oficial'
        
        params = {}
        
        if fecha_desde:
            params['fechaDesde'] = fecha_desde
        if fecha_hasta:
            params['fechaHasta'] = fecha_hasta
        
        data = self._make_request(self.ENDPOINTS[endpoint_key], params=params)
        
        if as_dataframe:
            return self._parse_estadisticas_cambiarias(data)
        
        return data
    
    def get_reservas(
        self,
        fecha_desde: Optional[str] = None,
        fecha_hasta: Optional[str] = None,
        as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get central bank reserves (international reserves).
        
        Reserves indicate:
            - Country's ability to defend currency
            - Import capacity
            - Macro stability
            
        Sharp declines in reserves correlate with financial crises.
        
        Args:
            fecha_desde: Start date 'YYYY-MM-DD'
            fecha_hasta: End date 'YYYY-MM-DD'
            as_dataframe: Return as DataFrame
            
        Returns:
            DataFrame with reserves time series
        """
        params = {}
        
        if fecha_desde:
            params['fechaDesde'] = fecha_desde
        if fecha_hasta:
            params['fechaHasta'] = fecha_hasta
        
        data = self._make_request(self.ENDPOINTS['reservas'], params=params)
        
        if as_dataframe:
            return self._parse_estadisticas_cambiarias(data)
        
        return data
    
    def get_tasas_interes(
        self,
        tipo: str = 'badlar',
        as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get interest rates (BADLAR, TNA, TEA, etc.).
        
        Interest rates are critical for:
            - Debt servicing capacity
            - Cost of capital
            - Financial distress indicators
        
        Args:
            tipo: Rate type ('badlar', 'tna', 'tea', 'tm20')
            as_dataframe: Return as DataFrame
            
        Returns:
            DataFrame with interest rate time series
        """
        params = {'tipo': tipo} if tipo else {}
        
        data = self._make_request(self.ENDPOINTS['tasas_interes'], params=params)
        
        if as_dataframe:
            return self._parse_tasas_interes(data)
        
        return data
    
    def get_agregados_monetarios(
        self,
        agregado: str = 'M2',
        as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get monetary aggregates (M1, M2, M3, etc.).
        
        Useful for macroeconomic context in distress analysis.
        
        Args:
            agregado: Agregado type ('M1', 'M2', 'M3')
            as_dataframe: Return as DataFrame
            
        Returns:
            DataFrame with monetary aggregate time series
        """
        params = {'agregado': agregado} if agregado else {}
        
        data = self._make_request(self.ENDPOINTS['agregados_monetarios'], params=params)
        
        if as_dataframe:
            return self._parse_agregados_monetarios(data)
        
        return data
    
    def fetch_company_distress_signals(
        self,
        cuit: str,
        include_cheques: bool = True,
        include_deuda: bool = True,
        days_back: int = 90,
        cheque_queries: Optional[List[Dict[str, int]]] = None
    ) -> Dict:
        """
        Fetch all distress signals for a specific company.
        
        This is the MAIN method for credit risk analysis. It combines:
            - Debt situation from Central de Deudores
            - Rejected checks from Cheques Denunciados
            - Macro context (optional)
        
        Args:
            cuit: Company CUIT (format: 'XX-XXXXXXXX-X')
            include_cheques: Whether to include rejected checks
            include_deuda: Whether to include debt information
            days_back: Days to look back for rejected checks
            cheque_queries: Optional list of real check queries. Each item must
                contain 'codigo_entidad' and 'numero_cheque'.
            
        Returns:
            Dictionary with distress indicators:
            - deuda_total: Total debt from all entities
            - situacion: Worst debt situation (1-5)
            - cheques_rechazados: Count of rejected checks
            - monto_cheques_rechazados: Total amount of rejected checks
            - riesgo: Risk score (0-100)
            
        Example:
            >>> client = BCRAAPIClient()
            >>> signals = client.fetch_company_distress_signals('30-12345678-9')
            >>> print(f"Risk Score: {signals['riesgo']}")
            >>> print(f"Rejected checks: {signals['cheques_rechazados']}")
        """
        distress_data = {
            'cuit': cuit,
            'fecha_consulta': datetime.now().strftime('%Y-%m-%d'),
            'deuda_total': 0,
            'situacion': 1,
            'cheques_rechazados': 0,
            'monto_cheques_rechazados': 0,
            'cheques_consultados': 0,
            'riesgo': 0
        }
        
        try:
            # Get debt information
            if include_deuda:
                deuda_df = self.get_central_deudores(cuit=cuit)
                
                if not deuda_df.empty:
                    # Total debt across all entities
                    distress_data['deuda_total'] = deuda_df['deuda'].sum()
                    
                    # Worst situation (highest number = worst)
                    distress_data['situacion'] = deuda_df['situacion'].max()
                    
                    logger.info(f"Debt info retrieved for {cuit}: Total={distress_data['deuda_total']}")
            
            # Get rejected checks
            if include_cheques:
                if not cheque_queries:
                    logger.info(
                        "Skipping cheques lookup for %s: the official API requires "
                        "codigo_entidad and numero_cheque for each consultation.",
                        cuit,
                    )
                else:
                    distress_data['cheques_consultados'] = len(cheque_queries)
                    for query in cheque_queries:
                        cheque_df = self.get_cheques_denunciados(
                            codigo_entidad=query['codigo_entidad'],
                            numero_cheque=query['numero_cheque']
                        )

                        if not cheque_df.empty and bool(cheque_df.iloc[0]['denunciado']):
                            distress_data['cheques_rechazados'] += 1

                    logger.info(
                        "Cheque queries for %s: %s checked, %s denounced",
                        cuit,
                        distress_data['cheques_consultados'],
                        distress_data['cheques_rechazados'],
                    )
            
            # Calculate risk score (0-100)
            distress_data['riesgo'] = self._calculate_risk_score(distress_data)
            
        except Exception as e:
            logger.error(f"Error fetching distress signals for {cuit}: {e}")
        
        return distress_data
    
    def _calculate_risk_score(self, distress_data: Dict) -> float:
        """
        Calculate composite risk score (0-100).
        
        Scoring methodology:
            - Situación (debt situation 1-5): 40% weight
            - Cheques rechazados: 30% weight
            - Monto cheques: 30% weight
        
        Higher score = Higher risk = Higher probability of distress
        
        Args:
            distress_data: Dictionary with debt and check information
            
        Returns:
            Risk score from 0 (healthy) to 100 (severe distress)
        """
        score = 0
        
        # Debt situation score (1=normal, 5=default)
        # Situation 1-2: Low risk (0-20 points)
        # Situation 3: Medium risk (30 points)
        # Situation 4-5: High risk (40-60 points)
        situacion = distress_data.get('situacion', 1)
        if situacion <= 2:
            situacion_score = situacion * 10
        elif situacion == 3:
            situacion_score = 30
        else:
            situacion_score = 40 + (situacion - 4) *20
        
        # Rejected checks count
        # 0 checks: 0 points
        # 1-2 checks: 10-20 points
        # 3-5 checks: 30-50 points
        # >5 checks: 60-80 points
        cheques = distress_data.get('cheques_rechazados', 0)
        if cheques == 0:
            cheques_score = 0
        elif cheques <= 2:
            cheques_score = cheques *10
        elif cheques <= 5:
            cheques_score = 30 + (cheques - 2) * 10
        else:
            cheques_score = min(80, 60 + (cheques - 5) * 5)
        
        # Total amount of rejected checks (relative)
        # This would need normalization based on company size
        # For now, use a simple threshold
        monto = distress_data.get('monto_cheques_rechazados', 0)
        if monto ==0:
            monto_score = 0
        elif monto <1000000:  # <1M pesos
            monto_score =10
        elif monto < 10000000:  # <10M pesos
            monto_score = 20
        else:
            monto_score = 30
        
        # Weighted average
        risk_score = (
            situacion_score *0.4 +
            cheques_score * 0.3 +
            monto_score * 0.3
        )
        
        return min(100, risk_score)
    
    # Parsing methods (convert JSON responses to DataFrames)
    
    def _parse_central_deudores(self, data: Dict) -> pd.DataFrame:
        """Parse Central de Deudores JSON response to DataFrame."""
        results = data.get('results', [])
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Standardize column names
        column_mapping = {
            'identificacion': 'cuit',
            'codigoEntidad': 'entidad',
            'situacionDeuda': 'situacion',
            'montoDeuda': 'deuda',
            'periodo': 'periodo'
        }
        
        df = df.rename(columns=column_mapping)
        
        logger.info(f"Parsed {len(df)} debt records")
        
        return df
    
    def _parse_cheques_denunciados(self, data: Dict) -> pd.DataFrame:
        """Parse Cheques Denunciados JSON response to a one-row DataFrame."""
        results = data.get('results', {})

        if not results:
            return pd.DataFrame()

        detalles = results.get('detalles', [])
        row = {
            'numero_cheque': results.get('numeroCheque'),
            'denunciado': results.get('denunciado', False),
            'fecha_procesamiento': results.get('fechaProcesamiento'),
            'denominacion_entidad': results.get('denominacionEntidad'),
            'cantidad_detalles': len(detalles),
            'causales': '; '.join(
                sorted({detalle.get('causal') for detalle in detalles if detalle.get('causal')})
            ),
            'detalles': detalles,
        }

        logger.info("Parsed cheque query for %s", row['numero_cheque'])

        return pd.DataFrame([row])

    def _parse_cheques_entidades(self, data: Dict) -> pd.DataFrame:
        """Parse the Cheques entities endpoint to DataFrame."""
        results = data.get('results', [])

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).rename(
            columns={
                'codigoEntidad': 'codigo_entidad',
                'denominacion': 'denominacion',
            }
        )
        logger.info("Parsed %s cheque entities", len(df))
        return df
    
    def _parse_estadisticas_cambiarias(self, data: Dict) -> pd.DataFrame:
        """Parse exchange rate statistics JSON response to DataFrame."""
        results = data.get('results', [])
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Standardize column names
        column_mapping = {
            'fecha': 'fecha',
            'compra': 'compra',
            'venta': 'venta',
            'promedio': 'promedio'
        }
        
        df = df.rename(columns=column_mapping)
        
        logger.info(f"Parsed {len(df)} exchange rate records")
        
        return df
    
    def _parse_tasas_interes(self, data: Dict) -> pd.DataFrame:
        """Parse interest rates JSON response to DataFrame."""
        results = data.get('results', [])
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        logger.info(f"Parsed {len(df)} interest rate records")
        
        return df
    
    def _parse_agregados_monetarios(self, data: Dict) -> pd.DataFrame:
        """Parse monetary aggregates JSON response to DataFrame."""
        results = data.get('results', [])
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        logger.info(f"Parsed {len(df)} monetary aggregate records")
        
        return df
    
    def close(self):
        """Close HTTP session."""
        self.session.close()
        logger.info("BCRAAPIClient session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_training_dataset(
    cuits: Optional[List[str]] = None,
    output_path: Union[str, Path] = 'data/processed/bcra_dataset.csv',
    include_macro: bool = True
) -> pd.DataFrame:
    """
    Create complete training dataset from BCRA APIs.
    
    This function fetches real data from multiple BCRA APIs and creates
    a production-ready dataset for financial distress prediction.
    
    Args:
        cuits: List of CUITs to query (default: all available)
        output_path: Where to save the final dataset
        include_macro: Whether to include macroeconomic context
        
    Returns:
        DataFrame ready for ML training
        
    Example:
        >>> # Fetch data for specific companies
        >>> cuits = ['30-12345678-9', '30-87654321-0']
        >>> df = create_training_dataset(cuits=cuits)
        >>> print(df.shape)
    """
    logger.info("=" * 80)
    logger.info("CREATING TRAINING DATASET FROM BCRA APIS")
    logger.info("=" * 80)
    
    client = BCRAAPIClient()
    
    all_records = []
    
    try:
        # Fetch debt information
        logger.info("Fetching debt information from Central de Deudores...")
        deuda_df = client.get_central_deudores()
        
        # Aggregate by CUIT
        if not deuda_df.empty:
            debt_by_cuit = deuda_df.groupby('cuit').agg({
                'deuda': 'sum',
                'situacion': 'max',
                'entidad': 'nunique'
            }).reset_index()
            debt_by_cuit.columns = ['cuit', 'deuda_total', 'peor_situacion', 'num_entidades']
        
        # Cheques Denunciados v1.0 does not expose bulk search by CUIT, so the
        # training dataset cannot enrich all companies with cheque signals unless
        # the calling code already has concrete (codigo_entidad, numero_cheque)
        # pairs to query individually.
        if not deuda_df.empty:
            dataset = debt_by_cuit
            dataset['cheques_rechazados'] =0
            dataset['monto_total_rechazado'] = 0
        else:
            logger.error("No data retrieved from BCRA APIs")
            return pd.DataFrame()
        
        # Fill NaN values
        dataset = dataset.fillna(0)
        
        # Calculate risk score for each company
        dataset['riesgo'] = dataset.apply(
            lambda row: client._calculate_risk_score(row.to_dict()),
            axis=1
        )
        
        # Create target variable (distress = 1 if risk score > 50)
        dataset['target'] = (dataset['riesgo'] > 50).astype(int)
        
        # Add macroeconomic context
        if include_macro:
            logger.info("Adding macroeconomic context...")
            
            # Get latest dollar rate
            dollar = client.get_dollar_rate(tipo='oficial')
            if not dollar.empty:
                latest_dollar = dollar.iloc[-1]['promedio']
                dataset['dollar_rate'] = latest_dollar
            
            # Get latest reserves
            reservas = client.get_reservas()
            if not reservas.empty:
                latest_reservas = reservas.iloc[-1]['valor']
                dataset['reservas_bcra'] = latest_reservas
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Shape: {dataset.shape}")
        logger.info(f"Columns: {list(dataset.columns)}")
        logger.info(f"Target distribution:\n{dataset['target'].value_counts()}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        return pd.DataFrame()
    
    finally:
        client.close()


def main():
    """Command-line interface for BCRA API client."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch financial data from BCRA APIs'
    )
    parser.add_argument(
        '--cuit',
        help='Specific CUIT to query'
    )
    parser.add_argument(
        '--output',
        default='data/processed/bcra_dataset.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--endpoint',
        choices=['deudores', 'cheques', 'cheques-entidades', 'dollar', 'reservas', 'tasas'],
        default='deudores',
        help='BCRA API endpoint to use'
    )
    parser.add_argument('--codigo-entidad', type=int, help='Entity code for Cheques API')
    parser.add_argument('--numero-cheque', type=int, help='Check number for Cheques API')
    
    args = parser.parse_args()
    
    client = BCRAAPIClient()
    
    try:
        if args.endpoint == 'deudores':
            df = client.get_central_deudores(cuit=args.cuit)
        elif args.endpoint == 'cheques':
            if args.codigo_entidad is None or args.numero_cheque is None:
                raise ValueError('--codigo-entidad and --numero-cheque are required for endpoint cheques')
            df = client.get_cheques_denunciados(
                codigo_entidad=args.codigo_entidad,
                numero_cheque=args.numero_cheque,
            )
        elif args.endpoint == 'cheques-entidades':
            df = client.get_cheques_entidades()
        elif args.endpoint == 'dollar':
            df = client.get_dollar_rate()
        elif args.endpoint == 'reservas':
            df = client.get_reservas()
        elif args.endpoint == 'tasas':
            df = client.get_tasas_interes()
        
        if not df.empty:
            print(df.head())
            print(f"\nTotal records: {len(df)}")
        else:
            print("No data retrieved")
        
    finally:
        client.close()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())