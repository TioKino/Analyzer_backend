"""
ShazamKit en Android (2026-09-30): el SDK pide un «developer token», un JWT
ES256 firmado con la clave Media Services de Apple Developer, que no puede ir
en la app. Lo firma Render (`shazam_token.py`) y el movil lo pide a
`GET /shazam/token`. Lo que se ata:
  - El token es un JWT que Apple puede verificar: {alg: ES256, kid} +
    {iss: team, iat, exp}, firmado r‖s con la clave.
  - Se reutiliza hasta que le queda una semana, y cambia si cambia la clave.
  - La clave pegada en una linea (con «\\n») vale igual.
  - Sin configurar, 503, y el movil sigue con AudD.
"""
import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from fastapi.testclient import TestClient

import shazam_token


def _pem():
    clave = ec.generate_private_key(ec.SECP256R1())
    pem = clave.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    return clave, pem


def _parte(b64):
    return base64.urlsafe_b64decode(b64 + '=' * (-len(b64) % 4))


@pytest.fixture
def configurado(monkeypatch):
    clave, pem = _pem()
    monkeypatch.setenv('SHAZAM_TEAM_ID', 'TEAM123456')
    monkeypatch.setenv('SHAZAM_KEY_ID', 'KEY1234567')
    monkeypatch.setenv('SHAZAM_PRIVATE_KEY', pem)
    shazam_token._cache.update(token=None, expira=0, huella=None)
    yield clave
    shazam_token._cache.update(token=None, expira=0, huella=None)


def test_el_token_es_un_jwt_que_apple_puede_verificar(configurado):
    t = shazam_token.token_de_shazam(reloj=lambda: 1_000_000)
    cab, cuerpo, firma = t['token'].split('.')
    assert json.loads(_parte(cab)) == {'alg': 'ES256', 'kid': 'KEY1234567'}
    c = json.loads(_parte(cuerpo))
    assert c == {'iss': 'TEAM123456', 'iat': 1_000_000,
                 'exp': 1_000_000 + shazam_token.DURACION}
    assert t['expira'] == c['exp']
    crudo = _parte(firma)
    assert len(crudo) == 64, 'JWS pide r‖s en crudo, no DER'
    der = encode_dss_signature(int.from_bytes(crudo[:32], 'big'),
                               int.from_bytes(crudo[32:], 'big'))
    configurado.public_key().verify(der, f'{cab}.{cuerpo}'.encode(),
                                    ec.ECDSA(hashes.SHA256()))
    assert shazam_token.DURACION <= 180 * 24 * 3600, 'Apple: 6 meses como mucho'


def test_se_reutiliza_hasta_que_le_queda_una_semana(configurado):
    t0 = 1_000_000
    a = shazam_token.token_de_shazam(reloj=lambda: t0)
    b = shazam_token.token_de_shazam(reloj=lambda: t0 + 3600)
    assert a == b
    casi = t0 + shazam_token.DURACION - shazam_token.RENOVAR + 1
    c = shazam_token.token_de_shazam(reloj=lambda: casi)
    assert c['token'] != a['token']
    assert c['expira'] == casi + shazam_token.DURACION


def test_otra_clave_otro_token(configurado, monkeypatch):
    a = shazam_token.token_de_shazam(reloj=lambda: 1_000_000)
    monkeypatch.setenv('SHAZAM_KEY_ID', 'OTRA123456')
    b = shazam_token.token_de_shazam(reloj=lambda: 1_000_000)
    assert json.loads(_parte(b['token'].split('.')[0]))['kid'] == 'OTRA123456'
    assert a['token'] != b['token']


def test_la_clave_pegada_en_una_linea_vale(configurado, monkeypatch):
    import os
    pem = os.environ['SHAZAM_PRIVATE_KEY']
    monkeypatch.setenv('SHAZAM_PRIVATE_KEY', pem.strip().replace('\n', '\\n'))
    assert shazam_token.token_de_shazam() is not None


def test_una_clave_rota_no_tumba_nada(configurado, monkeypatch):
    monkeypatch.setenv('SHAZAM_PRIVATE_KEY', 'esto no es un p8')
    assert shazam_token.token_de_shazam() is None


def test_el_endpoint(configurado):
    import main
    client = TestClient(main.app)
    r = client.get('/shazam/token')
    assert r.status_code == 200
    assert r.json()['token'].count('.') == 2


def test_sin_configurar_503(monkeypatch):
    for v in ('SHAZAM_TEAM_ID', 'SHAZAM_KEY_ID', 'SHAZAM_PRIVATE_KEY'):
        monkeypatch.delenv(v, raising=False)
    import main
    client = TestClient(main.app)
    assert client.get('/shazam/token').status_code == 503
