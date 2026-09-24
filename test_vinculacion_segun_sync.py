"""La vinculacion del movil, contada en sync.db y no en los eventos.

El paso `device_linked` del embudo de movil llevaba semanas en 3-4 % (6 de
177), y no habia forma de saber si eso media la vinculacion o la
instrumentacion: el evento solo lo emite el aparato que TECLEA el codigo, asi
que si se genera en el movil y se teclea en el ordenador, el movil vinculado no
cuenta. `sync.db` sabe quien comparte cuenta con quien, venga el codigo de
donde venga. Estos tests atan esa cuenta, y que sale AL LADO del numero de los
eventos para poder compararlos.
"""

import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

import routes.admin_panel as ap
from main import app

_SECRET = 'test-admin-secret-1234567890'


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return path


@pytest.fixture
def bases(monkeypatch):
    """Una cohorte de moviles con cada caso, y su sync.db."""
    analysis = _tmp_db()
    a = sqlite3.connect(analysis)
    a.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT, "
              "device_id TEXT, event_name TEXT, platform TEXT)")
    a.execute("CREATE TABLE device_first_seen (device_id TEXT PRIMARY KEY, "
              "first_day TEXT NOT NULL, first_platform TEXT, "
              "first_app_version TEXT)")
    aparatos = [
        ('m_pc_ok', 'ios'),        # vinculado a un PC y le llego biblioteca
        ('m_pc_vacio', 'android'),  # vinculado a un PC, sin biblioteca ajena
        ('m_solo', 'ios'),          # registrado, cuenta propia sin nadie
        ('m_nada', 'android'),      # nunca llego a sync
        ('m_dos_moviles', 'ios'),   # vinculado, pero con otro movil
        ('pc1', 'macos'),           # escritorio: fuera del embudo de movil
    ]
    for dev, plat in aparatos:
        a.execute("INSERT INTO events (timestamp, device_id, event_name, platform) "
                  "VALUES (datetime('now'), ?, 'app_opened', ?)", (dev, plat))
        a.execute("INSERT INTO device_first_seen (device_id, first_day, "
                  "first_platform) VALUES (?, date('now'), ?)", (dev, plat))
    # El evento solo lo emitio UNO: los demas tecleaban el codigo en el PC o
    # iban en una version sin el evento.
    a.execute("INSERT INTO events (timestamp, device_id, event_name, platform) "
              "VALUES (datetime('now'), 'm_pc_ok', 'device_linked', 'ios')")
    a.commit()
    a.close()

    sync = _tmp_db()
    s = sqlite3.connect(sync)
    s.executescript("""
        CREATE TABLE user_devices (device_id TEXT PRIMARY KEY, user_id TEXT,
            device_type TEXT, device_name TEXT, linked_at TEXT);
        CREATE TABLE sync_items (key TEXT PRIMARY KEY, data_type TEXT,
            item_key TEXT, payload TEXT, deleted INTEGER, updated_at TEXT,
            last_device_id TEXT, device_type TEXT, hash TEXT);
        CREATE TABLE device_seen (device_id TEXT, item_key TEXT, hash TEXT,
            payload TEXT, PRIMARY KEY (device_id, item_key));
    """)
    for dev, user, tipo in [
        ('m_pc_ok', 'u1', 'ios'), ('pc1', 'u1', 'macos'),
        ('m_pc_vacio', 'u2', 'android'), ('pc2', 'u2', 'windows'),
        ('m_solo', 'u3', 'ios'),
        ('m_dos_moviles', 'u5', 'ios'), ('otro_movil', 'u5', 'android'),
    ]:
        s.execute("INSERT INTO user_devices VALUES (?, ?, ?, '', datetime('now'))",
                  (dev, user, tipo))
    for key, tipo, origen in [
        ('u1|analysis|t1', 'analysis', 'pc1'),         # del PC: SI cuenta
        ('u2|analysis|t2', 'analysis', 'm_pc_vacio'),  # su propio eco: NO
        ('u1|library_folders|all', 'library_folders', 'pc1'),  # no es analisis
    ]:
        s.execute("INSERT INTO sync_items VALUES (?, ?, 'x', '{}', 0, "
                  "datetime('now'), ?, 'x', 'h')", (key, tipo, origen))
    for dev, key in [
        ('m_pc_ok', 'u1|analysis|t1'),
        ('m_pc_vacio', 'u2|analysis|t2'),
        ('m_pc_vacio', 'u1|library_folders|all'),
    ]:
        s.execute("INSERT INTO device_seen VALUES (?, ?, 'h', NULL)", (dev, key))
    s.commit()
    s.close()

    monkeypatch.setenv('DATABASE_PATH', analysis)
    monkeypatch.setenv('ADMIN_TOKEN', _SECRET)
    monkeypatch.setattr(ap, '_SYNC_DB_PATH', sync)
    yield analysis, sync
    for p in (analysis, sync):
        try:
            os.unlink(p)
        except OSError:
            pass


def _funnel(platform):
    r = TestClient(app).get('/admin/funnel', params={'platform': platform},
                            headers={'X-Admin-Secret': _SECRET})
    assert r.status_code == 200
    return r.json()


def test_cuenta_lo_que_sync_db_sabe(bases):
    v = _funnel('mobile')['vinculacion_segun_sync']
    assert v == {
        'cohorte': 5,
        'registrados': 4,            # m_nada nunca llego a sync
        'vinculados': 3,             # m_pc_ok, m_pc_vacio, m_dos_moviles
        'con_ordenador': 2,          # el de dos moviles no tiene PC
        'recibieron_biblioteca': 1,  # m_pc_vacio solo vio su propio eco
    }


def test_sale_al_lado_del_numero_de_los_eventos(bases):
    # Es el punto: los eventos dicen 1 y sync.db dice 3. Con los dos a la vista
    # se sabe cuanto del 3-4 % era instrumentacion.
    body = _funnel('mobile')
    paso = next(s for s in body['steps'] if s['event'] == 'device_linked')
    assert paso['devices'] == 1
    assert body['vinculacion_segun_sync']['vinculados'] == 3


def test_el_eco_propio_no_es_recibir_biblioteca(bases):
    # Un pull full=true tambien devuelve lo que subio el propio aparato. Si eso
    # contara, un movil que analizo tres temas solo saldria como «recibio
    # biblioteca» sin haber recibido nada de nadie.
    v = _funnel('mobile')['vinculacion_segun_sync']
    assert v['recibieron_biblioteca'] == 1


def test_solo_en_el_embudo_de_movil(bases):
    assert _funnel('desktop')['vinculacion_segun_sync'] is None


def test_sin_sync_db_el_embudo_sigue_entero(bases, monkeypatch):
    # Un agregado nuevo no puede ampliar lo que se rompe: si sync.db no esta,
    # se queda vacio el suyo y los pasos salen igual.
    monkeypatch.setattr(ap, '_SYNC_DB_PATH', '/no/existe/sync.db')
    body = _funnel('mobile')
    assert body['vinculacion_segun_sync'] is None
    assert body['steps'][0]['devices'] == 5
