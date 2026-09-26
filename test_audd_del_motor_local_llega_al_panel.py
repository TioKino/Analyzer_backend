"""Lo que gastan en AudD los motores locales llega al panel de Render.

El motor local (EXE de Windows y DMG) llama a AudD con su propio token al
analizar y en Escuchar, y lo apunta en SU `audd_call_log`. Hasta el 2026-09-26
eso no salía de la máquina: `by_source_30d`, el número con el que se decide el
precio de Pro, contaba de menos en la proporción de análisis locales.

Ahora cada motor manda a Render sus totales por día y vía, firmados, y el
panel los enseña al lado (`motor_local_30d`), sin tocar `by_source_30d` para
no partir la serie de `funnel_data/`.

    pytest test_audd_del_motor_local_llega_al_panel.py -v
"""

import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


@pytest.fixture
def app_mod(monkeypatch):
    import main

    monkeypatch.setattr(main, '_WRITE_AUTH_SECRET', 'test-write-secret')
    return main


def _enviar(app_mod, cuerpo, firmar=True):
    body = json.dumps(cuerpo).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if firmar:
        headers.update(app_mod._sign_write_payload(body))
    return TestClient(app_mod.app).post('/audd/motor-local', content=body,
                                        headers=headers)


def test_el_motor_cuenta_por_dia_y_via_sin_las_sesiones(db):
    for _ in range(3):
        db.log_audd_call('fp', True, source='analyze')
    db.log_audd_call('fp', False, source='analyze')
    db.log_audd_call('recognize', True, source='recognize')
    db.log_audd_call('recognize_session', True, source='recognize_session')
    cuentas = {c['source']: c for c in db.cuentas_audd_por_dia(time.time() - 86400)}
    assert set(cuentas) == {'analyze', 'recognize'}, 'una sesión no es una llamada'
    assert cuentas['analyze']['llamadas'] == 4
    assert cuentas['analyze']['aciertos'] == 3


def test_apuntar_una_llamada_avisa_si_hay_quien_escuche(db):
    avisos = []
    db.al_apuntar_audd = lambda: avisos.append(1)
    db.log_audd_call('fp', True, source='analyze')
    assert avisos == [1]


def test_sin_firma_no_entra(app_mod):
    r = _enviar(app_mod, {'motor_id': 'm' * 16, 'dias': []}, firmar=False)
    assert r.status_code == 401


def test_reenviar_sustituye_no_suma(app_mod):
    motor = uuid.uuid4().hex
    dia = {'dia': '2026-09-26', 'source': 'analyze', 'llamadas': 10, 'aciertos': 7}
    assert _enviar(app_mod, {'motor_id': motor, 'dias': [dia]}).status_code == 200
    dia['llamadas'] = 12
    _enviar(app_mod, {'motor_id': motor, 'dias': [dia]})
    fila = app_mod.db.conn.execute(
        'SELECT llamadas FROM audd_motor_local WHERE motor_id = ?',
        (motor,)).fetchone()
    assert fila[0] == 12
    app_mod.db.conn.execute('DELETE FROM audd_motor_local WHERE motor_id = ?', (motor,))
    app_mod.db.conn.commit()


def test_lo_que_no_cuadra_se_descarta(app_mod):
    motor = uuid.uuid4().hex
    r = _enviar(app_mod, {'motor_id': motor, 'dias': [
        {'dia': '2026-09-26', 'source': 'analyze', 'llamadas': 5},
        {'dia': 'ayer', 'source': 'analyze', 'llamadas': 5},
        {'dia': '2026-09-26', 'source': 'recognize_session', 'llamadas': 5},
        {'dia': '2026-09-26', 'source': 'identify', 'llamadas': 2, 'aciertos': 9},
    ]})
    assert r.json() == {'status': 'ok', 'dias': 1, 'descartados': 3}
    app_mod.db.conn.execute('DELETE FROM audd_motor_local WHERE motor_id = ?', (motor,))
    app_mod.db.conn.commit()


def test_el_resumen_suma_los_motores(db):
    hoy = time.strftime('%Y-%m-%d', time.gmtime())
    db.guardar_audd_motor_local('motor-a', [
        {'dia': hoy, 'source': 'analyze', 'llamadas': 10, 'aciertos': 6}])
    db.guardar_audd_motor_local('motor-b', [
        {'dia': hoy, 'source': 'analyze', 'llamadas': 5, 'aciertos': 5},
        {'dia': hoy, 'source': 'recognize', 'llamadas': 3, 'aciertos': 1}])
    db.guardar_audd_motor_local('motor-a', [
        {'dia': '2020-01-01', 'source': 'analyze', 'llamadas': 99}])
    r = db.resumen_audd_motor_local(30)
    assert r['by_source'] == {'analyze': 15, 'recognize': 3, 'identify': 0}
    assert r['total'] == 18 and r['aciertos'] == 12 and r['motores'] == 2


def test_el_motor_local_manda_sus_cuentas_firmadas(app_mod, monkeypatch):
    enviado = {}

    class _Resp:
        status_code = 200
        text = ''

    def _post(url, data, headers, timeout):
        enviado.update(url=url, cuerpo=json.loads(data), headers=headers)
        return _Resp()

    monkeypatch.setattr(app_mod, 'IS_LOCAL_ENGINE', True)
    monkeypatch.setattr(app_mod.requests, 'post', _post)
    monkeypatch.setattr(app_mod.db, 'cuentas_audd_por_dia', lambda desde: [
        {'dia': '2026-09-26', 'source': 'analyze', 'llamadas': 4, 'aciertos': 3}])
    assert app_mod._mandar_audd_a_render() is True
    assert enviado['url'].endswith('/audd/motor-local')
    assert 'X-Signature' in enviado['headers']
    assert enviado['cuerpo']['dias'][0]['llamadas'] == 4
    assert 8 <= len(enviado['cuerpo']['motor_id']) <= 64
    # Y el id del motor es estable entre envíos.
    assert app_mod._id_del_motor() == enviado['cuerpo']['motor_id']


def test_el_panel_lo_ensena_aparte():
    import routes.admin_panel as panel

    with open(panel.__file__, encoding='utf-8') as fh:
        fuente = fh.read()
    assert '"motor_local_30d": _resumen_audd_motor_local()' in fuente
