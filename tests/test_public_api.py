from unittest.mock import Mock

import pytest
import requests

from src import (
    CNVDataExtractor,
    DistressClassifier,
    DistressPredictor,
    FinancialRatioEngine,
    ModelEvaluator,
)
from src.data_acquisition.bcra_api import BCRAAPIClient
from src.features.ratio_calculator import main as ratio_calculator_main
from src.model import FinancialDataPreprocessor


def test_root_package_public_exports_are_importable():
    assert CNVDataExtractor.__name__ == 'CNVDataExtractor'
    assert FinancialRatioEngine.__name__ == 'FinancialRatioEngine'
    assert DistressClassifier.__name__ == 'DistressClassifier'
    assert DistressPredictor.__name__ == 'DistressPredictor'
    assert ModelEvaluator.__name__ == 'ModelEvaluator'


def test_model_package_public_exports_are_importable():
    assert DistressClassifier.__name__ == 'DistressClassifier'
    assert DistressPredictor.__name__ == 'DistressPredictor'
    assert FinancialDataPreprocessor.__name__ == 'FinancialDataPreprocessor'


def test_ratio_calculator_cli_module_is_importable():
    assert callable(ratio_calculator_main)


def test_bcra_client_caches_response_to_disk(tmp_path):
    client = BCRAAPIClient(cache_dir=tmp_path, cache_enabled=True, retry_attempts=1)
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {'results': [{'identificacion': '30-12345678-9'}]}
    client.session.get = Mock(return_value=response)

    data = client._make_request('/api/v1/test-endpoint')

    assert data == {'results': [{'identificacion': '30-12345678-9'}]}
    assert (tmp_path / '_api_v1_test-endpoint.json').exists()

    client.close()


def test_bcra_client_does_not_retry_permanent_404(tmp_path):
    client = BCRAAPIClient(cache_dir=tmp_path, cache_enabled=False, retry_attempts=3)
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError(
        '404 Client Error',
        response=Mock(status_code=404)
    )
    client.session.get = Mock(return_value=response)

    with pytest.raises(requests.HTTPError):
        client._make_request('/api/v1/missing-endpoint')

    assert client.session.get.call_count == 1

    client.close()


def test_bcra_cheques_query_uses_verified_v1_endpoint(tmp_path):
    client = BCRAAPIClient(cache_dir=tmp_path, cache_enabled=False, retry_attempts=1)
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        'status': 200,
        'results': {
            'numeroCheque': 20377516,
            'denunciado': True,
            'fechaProcesamiento': '2026-04-01',
            'denominacionEntidad': 'BANCO DE LA NACION ARGENTINA',
            'detalles': [{'sucursal': 524, 'numeroCuenta': 5240055962, 'causal': 'Denunciado por tercero'}],
        },
    }
    client.session.get = Mock(return_value=response)

    df = client.get_cheques_denunciados(codigo_entidad=11, numero_cheque=20377516)

    assert bool(df.iloc[0]['denunciado']) is True
    client.session.get.assert_called_once()
    assert client.session.get.call_args.args[0] == 'https://api.bcra.gob.ar/cheques/v1.0/denunciados/11/20377516'

    client.close()


def test_bcra_cheques_entities_parser_returns_expected_columns(tmp_path):
    client = BCRAAPIClient(cache_dir=tmp_path, cache_enabled=False, retry_attempts=1)
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        'status': 200,
        'results': [
            {'codigoEntidad': 11, 'denominacion': 'BANCO DE LA NACION ARGENTINA'},
            {'codigoEntidad': 17, 'denominacion': 'BANCO BBVA ARGENTINA S.A.'},
        ],
    }
    client.session.get = Mock(return_value=response)

    df = client.get_cheques_entidades()

    assert list(df.columns) == ['codigo_entidad', 'denominacion']
    assert len(df) == 2

    client.close()