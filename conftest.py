"""
Configuración de pytest para DJ ANALYZER
"""

import pytest
import sys
import os

# Añadir el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar variables de entorno para tests
os.environ.setdefault('AUDD_API_TOKEN', 'test_token')
os.environ.setdefault('DISCOGS_TOKEN', 'test_token')
os.environ.setdefault('BASE_URL', 'http://localhost:8000')


@pytest.fixture(scope="session")
def test_config():
    """Configuración global para tests"""
    return {
        'base_url': 'http://localhost:8000',
        'test_mode': True
    }


def pytest_configure(config):
    """Configuración inicial de pytest"""
    config.addinivalue_line(
        "markers", "slow: marca tests que tardan mucho"
    )
    config.addinivalue_line(
        "markers", "integration: tests de integración"
    )
    config.addinivalue_line(
        "markers", "unit: tests unitarios"
    )
