"""
Tests del contrato (track_data, processed_ok) de _send_to_audd, que gobierna el
FAN-OUT inteligente de /recognize:

  - AudD devuelve match           -> (dict, True)  -> exito, stop.
  - AudD procesa pero sin match   -> (None, True)  -> track NO en AudD; el caller
                                                      NO reintenta (seria en balde).
  - AudD no pudo fingerprintear   -> (None, False) -> audio malo; reintentar con
    (status != success)                              otro preprocesado SI vale.
  - HTTP error                    -> (None, False) -> idem, reintentar.

Asi el caso comun del DJ (tema underground que AudD no conoce: huella OK, sin
match) gasta 1 llamada en vez de 3.
"""

import os
import tempfile

import main


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _small_audio_file():
    f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    f.write(b'RIFF' + b'\x00' * 4096)  # <4MB -> no se recorta, se envia tal cual
    f.close()
    return f.name


def _run(monkeypatch, payload, status_code=200):
    path = _small_audio_file()
    monkeypatch.setattr(main.requests, 'post',
                        lambda *a, **k: _FakeResp(status_code, payload))
    try:
        return main._send_to_audd(path, 'tok', timeout=5)
    finally:
        os.unlink(path)


def test_match_returns_data_and_processed(monkeypatch):
    td, ok = _run(monkeypatch, {
        'status': 'success',
        'result': {'artist': 'Oxia', 'title': 'Domino'},
    })
    assert td == {'artist': 'Oxia', 'title': 'Domino'}
    assert ok is True


def test_success_no_match_is_processed_ok_true(monkeypatch):
    """El caso que motiva el ahorro: AudD fingerprinteo bien, sin match."""
    td, ok = _run(monkeypatch, {'status': 'success', 'result': None})
    assert td is None
    assert ok is True  # -> el caller NO reintenta


def test_audd_error_is_not_processed(monkeypatch):
    td, ok = _run(monkeypatch, {
        'status': 'error',
        'error': {'error_message': 'could not create fingerprint'},
    })
    assert td is None
    assert ok is False  # -> el caller SI reintenta con otro preprocesado


def test_http_error_is_not_processed(monkeypatch):
    td, ok = _run(monkeypatch, {}, status_code=500)
    assert td is None
    assert ok is False
