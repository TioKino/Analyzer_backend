"""El embudo con UNA definición (v3, `embudo.py`, 2026-11-25).

Cada test es uno de los fallos de la auditoría de ese día (`HISTORIAL.md` →
«2026-11-25 — El embudo definitivo»), con el caso que lo destapaba.

    pytest test_el_embudo_v3.py -v
"""

import json
import os
import sqlite3
import tempfile
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import embudo
from database import AnalysisDB

_SECRET = 'test-admin-secret-1234567890'
HOY = date(2026, 11, 25)


def _dia(hace):
    return (HOY - timedelta(days=hace)).isoformat()


class _Base:
    """Una analysis.db con el esquema de verdad y aparatos a mano."""

    def __init__(self):
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.db = AnalysisDB(self.path)

    def alta(self, dev, hace, plataforma, prueba=0, reidentificado=0):
        conn = self.db._open_conn()
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day, '
            'first_platform, prueba, reidentificado) VALUES (?,?,?,?,?)',
            (dev, _dia(hace), plataforma, prueba, reidentificado))
        conn.commit()
        conn.close()
        return self

    def ev(self, dev, hace, nombre, props=None, plataforma=None):
        conn = self.db._open_conn()
        conn.execute(
            'INSERT INTO events (timestamp, device_id, event_name, props, '
            'platform) VALUES (?,?,?,?,?)',
            (_dia(hace) + ' 12:00:00', dev, nombre,
             json.dumps(props) if props else None, plataforma))
        conn.commit()
        conn.close()
        return self

    def embudo(self, plataforma=None):
        conn = sqlite3.connect(self.path)
        try:
            return embudo.calcular(conn, hoy=HOY, plataforma=plataforma)
        finally:
            conn.close()

    def cerrar(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass


@pytest.fixture
def base():
    b = _Base()
    yield b
    b.cerrar()


def _pasos(e, grupo):
    return {p['paso']: p['llegaron'] for p in e['grupos'][grupo]['pasos']}


# ============================================================================
# 1. LA COHORTE MADURA: toda alta tiene el mismo tiempo para dar los pasos
# ============================================================================

def test_las_altas_de_los_ultimos_7_dias_no_entran(base):
    """Un alta de esta mañana no ha tenido tiempo de dar ningún paso: con la
    cohorte abierta, la conversión bajaba cuando llegaba más gente."""
    for dev, hace in (('hoy', 0), ('hace_7', 7), ('hace_8', 8),
                      ('hace_37', 37), ('hace_38', 38)):
        base.alta(dev, hace, 'windows').ev(dev, hace, 'app_opened')
    e = base.embudo()
    assert e['altas_desde'] == _dia(37) and e['altas_hasta'] == _dia(8)
    assert _pasos(e, 'escritorio')['alta'] == 2   # hace_8 y hace_37


def test_un_paso_pasados_7_dias_del_alta_no_cuenta(base):
    base.alta('tarde', 20, 'windows').ev('tarde', 20, 'app_opened')
    base.ev('tarde', 10, 'first_track_viewed')    # día 10: fuera
    assert _pasos(base.embudo(), 'escritorio')['valor'] == 0
    base.ev('tarde', 13, 'first_track_viewed')    # día 7: dentro
    assert _pasos(base.embudo(), 'escritorio')['valor'] == 1


# ============================================================================
# 2 y 3. EL PASO MÁS LEJANO: nunca sube, y un evento que falta no lo rompe
# ============================================================================

def test_quien_vio_su_musica_empezo_a_traerla_aunque_no_haya_evento(base):
    """«Vio su 1er track» superaba a «importó»: llegan por sync o por un XML.
    El paso más lejano cuenta a ese aparato en todos los anteriores."""
    base.alta('xml', 15, 'macos-dmg').ev('xml', 15, 'app_opened')
    base.ev('xml', 15, 'onboarding_completed')
    base.ev('xml', 15, 'first_track_viewed')
    pasos = _pasos(base.embudo(), 'escritorio')
    assert pasos['empezo'] == 1 and pasos['valor'] == 1
    directos = {p['paso']: p['directos']
                for p in base.embudo()['grupos']['escritorio']['pasos']}
    assert directos['empezo'] == 0, 'sin evento propio: se ve en directos'


def test_el_escritorio_sin_onboarding_shown_no_se_queda_a_cero(base):
    """El escritorio no mandaba `onboarding_shown`: su paso «Vio el
    onboarding» salía a 0 y el siguiente era mayor que él."""
    base.alta('pc', 15, 'windows').ev('pc', 15, 'app_opened')
    base.ev('pc', 15, 'onboarding_completed')
    pasos = _pasos(base.embudo(), 'escritorio')
    assert pasos['bienvenida_vista'] == 1
    assert pasos['bienvenida_pasada'] == 1


def test_ningun_paso_supera_al_anterior_ni_la_caida_es_negativa(base):
    casos = [('a', set()), ('b', {'app_opened'}),
             ('c', {'app_opened', 'first_track_viewed'}),
             ('d', {'import_started'}), ('e', {'onboarding_completed'})]
    for dev, eventos in casos:
        base.alta(dev, 12, 'windows')
        for ev in eventos:
            base.ev(dev, 12, ev)
    for g in ('escritorio', 'movil'):
        bloque = base.embudo()['grupos'][g]
        n = [p['llegaron'] for p in bloque['pasos']]
        assert n == sorted(n, reverse=True), g
        assert all(p['caen_pct'] is None or p['caen_pct'] >= 0
                   for p in bloque['pasos'])
    total = [p['llegaron'] for p in base.embudo()['todos']['pasos']]
    assert total == sorted(total, reverse=True)


# ============================================================================
# 4. «IMPORTÓ» SIN NADA DENTRO no es haber llegado a la música
# ============================================================================

def test_un_import_con_cero_temas_es_empezar_no_llegar(base):
    base.alta('cero', 15, 'windows').ev('cero', 15, 'app_opened')
    base.ev('cero', 15, 'import_started')
    base.ev('cero', 15, 'import_completed', {'ok': 0, 'errors': 30})
    e = base.embudo()
    pasos = _pasos(e, 'escritorio')
    assert pasos['empezo'] == 1 and pasos['valor'] == 0
    caminos = e['grupos']['escritorio']['caminos']
    assert caminos['import_con_temas'] == 0


def test_hechos_del_evento_lee_ok_del_escritorio_y_total_del_movil():
    assert 'import_con_temas' in embudo.hechos_del_evento(
        'import_completed', {'ok': 3}, '2026-11-01', '2026-11-01')
    assert 'import_con_temas' in embudo.hechos_del_evento(
        'import_completed', {'total': 2}, '2026-11-01', '2026-11-01')
    assert 'import_con_temas' not in embudo.hechos_del_evento(
        'import_completed', {'ok': 0}, '2026-11-01', '2026-11-01')


# ============================================================================
# 5. CADA APARATO CON LOS PASOS DE SU PLATAFORMA; el total suma los dos
# ============================================================================

def test_en_el_movil_identificar_un_tema_es_llegar_al_valor(base):
    base.alta('m', 15, 'ios').ev('m', 15, 'app_opened')
    base.ev('m', 15, 'onboarding_shown').ev('m', 15, 'onboarding_completed')
    base.ev('m', 15, 'listen_result', {'outcome': 'found'})
    assert _pasos(base.embudo(), 'movil')['valor'] == 1


def test_en_el_movil_escuchar_sin_acierto_es_empezar(base):
    base.alta('m', 15, 'android').ev('m', 15, 'app_opened')
    base.ev('m', 15, 'listen_result', {'outcome': 'no_match'})
    pasos = _pasos(base.embudo(), 'movil')
    assert pasos['empezo'] == 1 and pasos['valor'] == 0


def test_el_total_suma_cada_aparato_con_sus_pasos(base):
    base.alta('pc', 15, 'windows').ev('pc', 15, 'first_track_viewed')
    base.alta('m', 15, 'ios').ev('m', 15, 'listen_result', {'outcome': 'found'})
    e = base.embudo()
    total = {p['paso']: p['llegaron'] for p in e['todos']['pasos']}
    assert total['alta'] == 2 and total['valor'] == 2


def test_la_plataforma_es_la_del_alta(base):
    for dev, plat in (('mas', 'macos-mas'), ('dmg', 'macos-dmg'),
                      ('viejo', 'macos'), ('win', 'windows'),
                      ('ios', 'ios'), ('and', 'android')):
        base.alta(dev, 15, plat)
    e = base.embudo()
    assert e['grupos']['escritorio']['cohorte'] == 4, 'los TRES Mac'
    assert e['grupos']['movil']['cohorte'] == 2
    assert base.embudo('macos-mas')['todos']['cohorte'] == 1


def test_sin_plataforma_se_cuenta_aparte_y_no_cae_en_un_grupo(base):
    base.alta('raro', 15, None)
    e = base.embudo()
    assert e['sin_plataforma'] == 1
    assert e['todos']['cohorte'] == 0


def test_sin_plataforma_en_el_alta_vale_la_del_primer_evento(base):
    base.alta('sembrado', 15, None).ev('sembrado', 15, 'app_opened',
                                       plataforma='android')
    assert base.embudo()['grupos']['movil']['cohorte'] == 1


# ============================================================================
# LOS QUE NO SON ALTAS: prueba y reidentificados
# ============================================================================

def test_los_aparatos_de_prueba_y_los_reidentificados_quedan_fuera(base):
    base.alta('real', 15, 'windows')
    base.alta('flutter_run', 15, 'macos-mas', prueba=1)
    base.alta('borro_sus_datos', 15, 'ios', reidentificado=1)
    e = base.embudo()
    assert e['todos']['cohorte'] == 1
    assert e['excluidos'] == {'prueba': 1, 'reidentificado': 1}


def test_log_event_deja_la_marca_puesta_para_siempre(base):
    base.db.log_event(device_id='dev', event_name='app_opened',
                      platform='macos-mas', prueba=True)
    base.db.log_event(device_id='dev', event_name='session_start',
                      platform='macos-mas')
    conn = base.db._open_conn()
    fila = conn.execute('SELECT prueba, reidentificado FROM device_first_seen '
                        "WHERE device_id = 'dev'").fetchone()
    conn.close()
    assert tuple(fila) == (1, 0)


def test_log_event_devuelve_el_id_del_evento_y_0_si_no_se_guarda(base):
    assert base.db.log_event(device_id='d', event_name='app_opened') > 0
    conn = base.db._open_conn()
    conn.execute('DROP TABLE events')
    conn.commit()
    conn.close()
    assert base.db.log_event(device_id='d', event_name='app_opened') == 0


# ============================================================================
# LA VUELTA: la misma cohorte que la retención reciente, y por valor
# ============================================================================

def test_la_vuelta_del_embudo_es_la_d7_de_la_retencion_reciente(base):
    """Dos números para lo mismo serían otro error: el embudo y la retención
    reciente miden la vuelta en 7 días sobre la MISMA cohorte."""
    for i, (hace, vuelve) in enumerate(((10, 7), (20, None), (30, 25),
                                        (36, 35), (5, 3), (45, 40))):
        dev = f'd{i}'
        base.alta(dev, hace, 'ios' if i % 2 else 'windows')
        base.ev(dev, hace, 'app_opened')
        if vuelve is not None:
            base.ev(dev, vuelve, 'session_start')
    e = base.embudo()
    rec = base.db.get_return_rates_recientes(hoy=HOY)
    assert e['todos']['volvio_7d']['todos']['cohorte'] == rec['total']['d7']['cohort']
    assert e['todos']['volvio_7d']['todos']['volvieron'] == rec['total']['d7']['returned']
    for g in ('escritorio', 'movil'):
        assert (e['grupos'][g]['volvio_7d']['todos']['volvieron']
                == rec['por_grupo'][g]['d7']['returned']), g


def test_la_vuelta_se_parte_por_haber_llegado_al_valor(base):
    base.alta('activo', 15, 'windows').ev('activo', 15, 'first_track_viewed')
    base.ev('activo', 12, 'session_start')
    base.alta('perdido', 15, 'windows').ev('perdido', 15, 'app_opened')
    v = base.embudo()['grupos']['escritorio']['volvio_7d']
    assert v['con_valor'] == {'cohorte': 1, 'volvieron': 1, 'pct': 100.0}
    assert v['sin_valor'] == {'cohorte': 1, 'volvieron': 0, 'pct': 0.0}


def test_el_mismo_dia_del_alta_no_es_volver(base):
    base.alta('d', 15, 'ios').ev('d', 15, 'app_opened')
    base.ev('d', 15, 'session_start')
    assert base.embudo()['todos']['volvio_7d']['todos']['volvieron'] == 0


# ============================================================================
# EL PANEL: /admin/funnel y /client-event
# ============================================================================

@pytest.fixture
def panel(monkeypatch, base):
    monkeypatch.setenv('DATABASE_PATH', base.path)
    monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
    return base


def test_el_panel_da_los_pasos_con_la_forma_de_siempre(panel):
    from main import app
    from datetime import datetime, timezone
    hoy = datetime.now(timezone.utc).date()
    conn = panel.db._open_conn()
    conn.execute("INSERT INTO device_first_seen (device_id, first_day, "
                 "first_platform) VALUES ('pc', ?, 'windows')",
                 ((hoy - timedelta(days=10)).isoformat(),))
    conn.commit()
    conn.close()
    r = TestClient(app).get('/admin/funnel', params={'platform': 'desktop'},
                            headers={'X-Admin-Secret': _SECRET})
    assert r.status_code == 200
    b = r.json()
    assert [s['event'] for s in b['steps']] == list(embudo.PASOS)
    assert set(b['steps'][0]) == {'event', 'label', 'devices', 'pct_of_top',
                                  'drop_from_prev'}
    assert b['steps'][0]['devices'] == 1 and b['cohort_size'] == 1
    assert b['embudo']['definicion'] == embudo.DEFINICION


def test_client_event_dice_logged_false_si_no_se_guardo(monkeypatch):
    """`log_event` se traga el `OperationalError` (BD bloqueada) y devuelve 0;
    hasta el 2026-11-25 el endpoint contestaba `logged: true` igual, y el
    cliente no lo contaba como perdido."""
    import main
    monkeypatch.setattr(main.db, 'log_event', lambda **kw: 0)
    r = TestClient(main.app).post('/client-event',
                                  json={'event_name': 'app_opened'},
                                  headers={'X-Device-Id': 'd'})
    assert r.status_code == 202
    assert r.json()['logged'] is False


def test_client_event_pasa_las_marcas(monkeypatch):
    import main
    visto = {}
    monkeypatch.setattr(main.db, 'log_event',
                        lambda **kw: visto.update(kw) or 7)
    r = TestClient(main.app).post(
        '/client-event',
        json={'event_name': 'app_opened', 'prueba': True,
              'reidentificado': False},
        headers={'X-Device-Id': 'd'})
    assert r.json()['logged'] is True
    assert visto['prueba'] is True and visto['reidentificado'] is False


# ============================================================================
# EL TOTAL NO TIENE OTRA DEFINICIÓN ESCONDIDA EN EL PANEL
# ============================================================================

def test_el_panel_no_tiene_listas_de_pasos_propias():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'routes', 'admin_panel.py'), encoding='utf-8').read()
    assert '_FUNNEL_DEFAULT = [' not in src
    assert '_FUNNEL_MOBILE = [' not in src
    assert 'embudo.calcular(' in src


def test_sin_base_de_datos_salen_los_seis_pasos_a_cero(monkeypatch):
    """Un paso sin datos sale a 0, no desaparece. El CI lo cazó el 25-nov:
    un test anterior dejaba `DATABASE_PATH` en una base ya borrada y el
    embudo nuevo contestaba sin pasos (el de antes daba los seis a cero)."""
    from main import app
    monkeypatch.setenv('DATABASE_PATH', '/no/existe/analysis.db')
    monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
    for plataforma in (None, 'desktop', 'mobile'):
        params = {'platform': plataforma} if plataforma else {}
        b = TestClient(app).get('/admin/funnel', params=params,
                                headers={'X-Admin-Secret': _SECRET}).json()
        assert [s['event'] for s in b['steps']] == list(embudo.PASOS), plataforma
        assert all(s['devices'] == 0 for s in b['steps'])
        assert b['cohort_size'] == 0
        assert b['embudo']['todos']['cohorte'] == 0, 'sin base: vacio, no fallo'


def test_si_el_calculo_falla_los_pasos_salen_y_el_fallo_se_dice(monkeypatch, panel):
    from main import app

    def _revienta(*a, **kw):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(embudo, 'calcular', _revienta)
    b = TestClient(app).get('/admin/funnel', headers={'X-Admin-Secret': _SECRET}).json()
    assert len(b['steps']) == len(embudo.PASOS)
    assert b['embudo'] is None, '«no esta» y «fallo» no dicen lo mismo'
