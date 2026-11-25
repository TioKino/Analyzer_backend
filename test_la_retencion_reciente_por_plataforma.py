"""
La retencion de las cohortes RECIENTES, por plataforma
(`get_return_rates_recientes`, `returns_recientes` de /admin/retention).

La rolling de siempre (`get_return_rates`) baja sola: el D0 sale de
`device_first_seen`, que no se purga, y el «volvio» de `events`, que se purga
a los 90 dias. Un veterano que volvio pierde sus eventos y sigue en el
denominador. Lectura 61 (17-nov): D7 21,1 → 17,5 % con ~220 aparatos nuevos
y los retenidos clavados en ~220. Esta medida solo mira las cohortes cuyos
eventos siguen ahi, y las parte por plataforma.
"""

import os
import tempfile
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'ret.db'))


def _dia(days_ago):
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime('%Y-%m-%d')


def _alta(db, device_id, days_ago, platform):
    conn = db._open_conn()
    conn.execute(
        'INSERT INTO device_first_seen (device_id, first_day, first_platform) '
        'VALUES (?,?,?)', (device_id, _dia(days_ago), platform))
    conn.commit()
    conn.close()


def _ev(db, device_id, days_ago, name='app_opened'):
    ts = (datetime.utcnow() - timedelta(days=days_ago)).strftime(
        '%Y-%m-%d %H:%M:%S')
    conn = db._open_conn()
    conn.execute(
        'INSERT INTO events (timestamp, device_id, event_name) VALUES (?,?,?)',
        (ts, device_id, name))
    conn.commit()
    conn.close()


def test_sin_datos_no_revienta():
    out = _db().get_return_rates_recientes()
    assert out['total']['d7'] == {'cohort': 0, 'returned': 0, 'rate': 0.0}
    assert out['por_plataforma'] == {}
    assert out['d7_por_semana'] == []


def test_el_veterano_purgado_no_entra():
    """El caso que motiva esto: un aparato de hace 200 dias que volvio, pero
    cuyos eventos ya se purgaron. En la rolling de siempre cuenta como «no
    volvio»; en la reciente no esta."""
    db = _db()
    _alta(db, 'veterano', 200, 'ios')
    _alta(db, 'nuevo', 20, 'ios')
    _ev(db, 'nuevo', 20)
    _ev(db, 'nuevo', 17)
    vieja = db.get_return_rates()['d7']
    assert vieja['cohort'] == 2 and vieja['returned'] == 1   # 50 %: el sesgo
    reciente = db.get_return_rates_recientes()['total']['d7']
    assert reciente == {'cohort': 1, 'returned': 1, 'rate': 100.0}


def test_reparte_por_plataforma_y_por_grupo():
    db = _db()
    _alta(db, 'i1', 30, 'ios')
    _ev(db, 'i1', 28)                       # volvio
    _alta(db, 'a1', 30, 'android')          # no volvio
    _alta(db, 'w1', 30, 'windows')
    _ev(db, 'w1', 25)
    _alta(db, 'm1', 30, 'macos-mas')
    _alta(db, 'm2', 30, 'macos-dmg')
    _ev(db, 'm2', 29, 'session_start')
    _alta(db, 'm3', 30, 'macos')            # cliente anterior a 2.9.11
    _alta(db, 'x', 30, None)
    out = db.get_return_rates_recientes()
    pp = out['por_plataforma']
    assert pp['ios']['d7']['returned'] == 1
    assert pp['android']['d7'] == {'cohort': 1, 'returned': 0, 'rate': 0.0}
    assert pp['sin']['d7']['cohort'] == 1
    g = out['por_grupo']
    assert g['movil']['d7'] == {'cohort': 2, 'returned': 1, 'rate': 50.0}
    # Los TRES nombres del Mac son escritorio.
    assert g['escritorio']['d7']['cohort'] == 4
    assert g['escritorio']['d7']['returned'] == 2
    assert g['sin']['d7']['cohort'] == 1
    assert out['total']['d7']['cohort'] == 7
    assert out['total']['d7']['returned'] == 3


def test_cada_dn_mide_sus_30_dias_de_altas_con_edad_para_medir():
    """Desde el 2026-11-25 (`embudo.ventana`): para cada DN, los 30 dias de
    altas que ya han tenido N dias enteros. D1 = altas de hace 2 a 31 dias,
    D7 = de 8 a 37 (la cohorte del embudo), D28 = de 29 a 58. Quien llego hace
    3 dias no entra en D7: saldria como perdido sin haber tenido tiempo."""
    db = _db()
    _alta(db, 'hace_3', 3, 'ios')
    _alta(db, 'hace_10', 10, 'ios')
    _alta(db, 'hace_40', 40, 'ios')
    _alta(db, 'hace_70', 70, 'ios')
    out = db.get_return_rates_recientes()['total']
    assert out['d1']['cohort'] == 2      # 3 y 10
    assert out['d7']['cohort'] == 1      # 10
    assert out['d28']['cohort'] == 1     # 40
    assert out['d28']['cohort'] + out['d7']['cohort'] < 4, 'el de 70 no entra'


def test_la_vuelta_fuera_de_la_ventana_no_cuenta():
    db = _db()
    _alta(db, 'd', 30, 'android')
    _ev(db, 'd', 30)                         # el mismo D0 no es volver
    _ev(db, 'd', 20)                         # dia 10: fuera de D7
    out = db.get_return_rates_recientes()['total']
    assert out['d7'] == {'cohort': 1, 'returned': 0, 'rate': 0.0}
    assert out['d28']['returned'] == 1


def test_el_d7_por_semana_va_por_el_lunes_del_alta():
    db = _db()
    for i, dias in enumerate((20, 21, 22, 23, 24, 25, 26, 27)):
        _alta(db, f'd{i}', dias, 'ios')
    _ev(db, 'd0', 18)
    semanas = db.get_return_rates_recientes()['d7_por_semana']
    assert sum(s['cohort'] for s in semanas) == 8
    assert sum(s['returned'] for s in semanas) == 1
    for s in semanas:
        assert datetime.strptime(s['semana'], '%Y-%m-%d').weekday() == 0
    assert [s['semana'] for s in semanas] == sorted(s['semana'] for s in semanas)


def test_el_panel_la_devuelve(monkeypatch):
    from main import app
    secreto = 'test-admin-secret-1234567890'
    monkeypatch.setenv('ADMIN_TOKEN', secreto)
    r = TestClient(app).get('/admin/retention',
                            headers={'X-Admin-Secret': secreto})
    assert r.status_code == 200
    rr = r.json()['returns_recientes']
    assert rr is not None
    assert {'ventana_dias', 'total', 'por_grupo', 'por_plataforma',
            'd7_por_semana'} <= set(rr)
