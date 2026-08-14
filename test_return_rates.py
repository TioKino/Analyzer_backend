"""
RETENCION ROLLING (get_return_rates) — "volvio ALGUNA vez en N dias".

Complementa el D1/D7/D28 estricto (criterio App Store: "abrio EXACTAMENTE el
dia N"). El caso que motiva esto esta en test_estricto_vs_rolling: un usuario
que vuelve los dias 3 y 5 cuenta como PERDIDO en el D7 estricto pero SI volvio.
Con cohortes pequeñas esa diferencia cambia una decision de producto (activar o
posponer el paywall), asi que se miden las dos.
"""

import os
import tempfile
from datetime import datetime, timedelta

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'ret.db'))


def _ts(days_ago):
    """Timestamp ISO de hace N dias (la columna `day` se deriva de aqui)."""
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime(
        '%Y-%m-%d %H:%M:%S')


def _ev(db, device_id, event_name, days_ago):
    conn = db._open_conn()
    conn.execute(
        'INSERT INTO events (timestamp, device_id, event_name) VALUES (?,?,?)',
        (_ts(days_ago), device_id, event_name))
    conn.commit()
    conn.close()


def test_sin_datos_devuelve_ceros():
    db = _db()
    out = db.get_return_rates()
    assert out['d7']['cohort'] == 0
    assert out['d7']['rate'] == 0.0


def test_cuenta_a_quien_volvio_dentro_de_la_ventana():
    db = _db()
    _ev(db, 'devA', 'app_opened', 10)   # instalo hace 10 dias
    _ev(db, 'devA', 'app_opened', 7)    # volvio al dia 3
    out = db.get_return_rates()
    assert out['d7']['cohort'] == 1
    assert out['d7']['returned'] == 1
    assert out['d7']['rate'] == 100.0


def test_no_cuenta_a_quien_nunca_volvio():
    db = _db()
    _ev(db, 'devA', 'app_opened', 10)
    out = db.get_return_rates()
    assert out['d7']['cohort'] == 1
    assert out['d7']['returned'] == 0


def test_session_start_tambien_cuenta_como_vuelta():
    """Volver a primer plano sin arranque en frio TAMBIEN es volver."""
    db = _db()
    _ev(db, 'devA', 'app_opened', 10)
    _ev(db, 'devA', 'session_start', 8)
    out = db.get_return_rates()
    assert out['d7']['returned'] == 1


def test_actividad_del_mismo_dia_0_no_cuenta():
    """Abrir varias veces el dia de la instalacion NO es retencion."""
    db = _db()
    _ev(db, 'devA', 'app_opened', 10)
    _ev(db, 'devA', 'session_start', 10)  # mismo dia D0
    out = db.get_return_rates()
    assert out['d7']['returned'] == 0


def test_vuelta_fuera_de_la_ventana_no_cuenta_en_d7():
    db = _db()
    _ev(db, 'devA', 'app_opened', 30)
    _ev(db, 'devA', 'app_opened', 10)  # dia 20: fuera de D7, dentro de D28
    out = db.get_return_rates()
    assert out['d7']['returned'] == 0
    assert out['d28']['returned'] == 1


def test_cohorte_excluye_devices_demasiado_nuevos():
    """Quien instalo ayer no puede evaluarse a 7 dias."""
    db = _db()
    _ev(db, 'nuevo', 'app_opened', 1)
    out = db.get_return_rates()
    assert out['d7']['cohort'] == 0
    assert out['d1']['cohort'] == 1


def test_estricto_vs_rolling():
    """EL CASO QUE MOTIVA ESTA METRICA.

    Un DJ instala hace 10 dias y vuelve los dias 3 y 5 — pero NO el dia 7
    exacto. El criterio App Store lo da por perdido; el rolling ve que si
    volvio. Con cohortes pequeñas esa diferencia es la que decide si el
    problema es 'no vuelven' o 'no vuelven justo el septimo dia'.
    """
    db = _db()
    _ev(db, 'devA', 'app_opened', 10)  # D0
    _ev(db, 'devA', 'app_opened', 7)   # dia 3
    _ev(db, 'devA', 'app_opened', 5)   # dia 5

    estricto = db.get_retention_cohorts()
    rolling = db.get_return_rates()

    assert estricto['d7']['retained'] == 0, 'no abrio el dia 7 exacto'
    assert rolling['d7']['returned'] == 1, 'pero SI volvio en la primera semana'


def test_varios_devices_ratio_correcto():
    db = _db()
    for d in ('a', 'b', 'c', 'd'):
        _ev(db, d, 'app_opened', 10)
    _ev(db, 'a', 'app_opened', 8)  # solo 'a' vuelve
    out = db.get_return_rates()
    assert out['d7']['cohort'] == 4
    assert out['d7']['returned'] == 1
    assert out['d7']['rate'] == 25.0


def test_endpoint_devuelve_ambas_metricas():
    """Wiring HTTP: /admin/retention debe traer `cohorts` Y `returns`."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    secret = os.getenv('ADMIN_TOKEN') or os.getenv('ADMIN_SECRET') or ''
    r = client.get('/admin/retention', headers={'X-Admin-Secret': secret})
    if r.status_code == 401:
        return  # sin token configurado en el entorno: el resto ya lo cubre
    assert r.status_code == 200
    body = r.json()
    assert 'cohorts' in body
    assert 'returns' in body


# ── Metricas de HERRAMIENTA (uso a rafagas, no diario) ──────────────────

def test_ever_returned_cuenta_vuelta_muy_tardia():
    """EL PATRON REAL DEL OWNER: importa su musica, no vuelve en semanas, y
    reaparece cuando tiene un rato (vacaciones). D7/D28 lo dan por perdido;
    `ever_returned` ve que sigue vivo."""
    db = _db()
    _ev(db, 'devA', 'app_opened', 90)   # instalo hace 3 meses
    _ev(db, 'devA', 'app_opened', 5)    # reaparecio hace 5 dias
    out = db.get_tool_usage_metrics()
    assert out['ever_returned']['cohort'] == 1
    assert out['ever_returned']['returned'] == 1
    # Y con criterio de app social habria contado como perdido:
    assert db.get_return_rates()['d7']['returned'] == 0


def test_ever_returned_no_cuenta_al_que_solo_abrio_una_vez():
    db = _db()
    _ev(db, 'devA', 'app_opened', 30)
    out = db.get_tool_usage_metrics()
    assert out['ever_returned']['returned'] == 0


def test_ever_returned_ignora_instalaciones_de_ayer():
    """Quien instalo ayer aun no ha tenido ocasion de volver."""
    db = _db()
    _ev(db, 'nuevo', 'app_opened', 0)
    out = db.get_tool_usage_metrics()
    assert out['ever_returned']['cohort'] == 0


def test_active_last_30d():
    db = _db()
    _ev(db, 'vivo', 'app_opened', 10)
    _ev(db, 'zombi', 'app_opened', 200)
    out = db.get_tool_usage_metrics()
    assert out['active_last_30d'] == 1


def test_deep_users_cuenta_quien_importo():
    """Un usuario que importo 3.000 tracks y vuelve cada dos meses vale mas
    que diez que abrieron y se fueron."""
    db = _db()
    _ev(db, 'a', 'app_opened', 20)
    _ev(db, 'a', 'import_completed', 20)
    _ev(db, 'b', 'app_opened', 20)  # abrio pero nunca importo
    out = db.get_tool_usage_metrics()
    assert out['deep_users'] == 1


def test_mediana_entre_visitas():
    """Si sale ~30, el producto es MENSUAL y las metricas diarias sobran."""
    db = _db()
    _ev(db, 'a', 'app_opened', 60)
    _ev(db, 'a', 'app_opened', 50)   # hueco de 10
    _ev(db, 'a', 'app_opened', 40)   # hueco de 10
    out = db.get_tool_usage_metrics()
    assert out['median_days_between_visits'] == 10.0


def test_mediana_none_si_nadie_repite():
    db = _db()
    _ev(db, 'a', 'app_opened', 10)
    out = db.get_tool_usage_metrics()
    assert out['median_days_between_visits'] is None


def test_ventanas_largas_en_return_rates():
    db = _db()
    _ev(db, 'a', 'app_opened', 100)
    _ev(db, 'a', 'app_opened', 50)  # dia 50: fuera de D28, dentro de D60/D90
    out = db.get_return_rates()
    assert out['d28']['returned'] == 0
    assert out['d60']['returned'] == 1
    assert out['d90']['returned'] == 1
