"""Las rejillas de la comunidad se piden por LOTE (auditoría del flujo de
análisis, tanda 3, 2026-10-06).

El import de escritorio pedía `GET /community/beat-grid/{huella}` por cada
tema nuevo, sin esperar a ninguno: con el pre-check acertando, cientos seguidos
contra Render. Ahora un `POST /community/beat-grid/batch` devuelve solo las
validadas (tres cuentas que coinciden), y el cálculo por cuenta se hace solo
para las huellas que tienen alguna corrección.

    pytest test_la_rejilla_de_la_comunidad_por_lote.py -v
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _tres(db, fp, ajuste=0.10):
    for aparato in ('a', 'b', 'c'):
        db.submit_beat_grid_correction(fp, aparato, ajuste, 0.010, 128.0)


def test_solo_salen_las_validadas(db):
    buena, una_sola, nada = (uuid.uuid4().hex for _ in range(3))
    _tres(db, buena)
    db.submit_beat_grid_correction(una_sola, 'a', 0.2, 0.0, 128.0)
    r = db.get_community_beat_grids([buena, una_sola, nada])
    assert set(r) == {buena}
    assert r[buena]['validated'] is True
    assert r[buena]['bpm_adjust'] == pytest.approx(0.10)
    # Lo mismo que da la consulta de uno en uno.
    assert r[buena] == db.get_community_beat_grid(buena)


def test_el_calculo_caro_solo_para_las_que_tienen_algo(db, monkeypatch):
    buena = uuid.uuid4().hex
    _tres(db, buena)
    calculadas = []
    original = db.get_community_beat_grid

    def contando(fp):
        calculadas.append(fp)
        return original(fp)

    monkeypatch.setattr(db, 'get_community_beat_grid', contando)
    db.get_community_beat_grids([buena] + [uuid.uuid4().hex for _ in range(300)])
    assert calculadas == [buena]


def test_el_endpoint(monkeypatch):
    import main
    buena = uuid.uuid4().hex
    _tres(main.db, buena)
    client = TestClient(main.app)
    r = client.post('/community/beat-grid/batch',
                    json={'fingerprints': [buena, uuid.uuid4().hex]})
    assert r.status_code == 200
    assert list(r.json()['rejillas']) == [buena]
    assert client.post('/community/beat-grid/batch',
                       json={'fingerprints': ['x'] * 501}).status_code == 400
