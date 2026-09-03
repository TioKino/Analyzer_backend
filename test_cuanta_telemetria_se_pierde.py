"""Todos los numeros del embudo son una cota inferior, y nadie sabia de cuanto.

`EventReporter.log` es un `http.post` fire-and-forget con timeout de 5 s: lo que
se dispara sin red se pierde y no queda ni rastro. La nota que habia en PENDING
decia —con razon— que **medir cuanto se pierde es mas barato que construir la
cola offline a ciegas**. Esto es la medida.

Y al montarla salio que las causas son DOS, no una:

  `sin_red`    la peticion ni salio. Lo arregla una cola offline.
  `rechazado`  llego y el servidor NO lo guardo. `/client-event` devuelve
               `202 {"logged": false}` cuando el INSERT falla, y el cliente
               daba por bueno cualquier HTTP. Durante el bug de los NULL en
               `device_first_seen` los eventos de la web se cayeron asi:
               invisibles desde los dos lados a la vez. Aqui no hay cola que
               valga — hay un bug en el servidor.

Piden arreglos opuestos, asi que no pueden compartir contador.

    pytest test_cuanta_telemetria_se_pierde.py -v
"""

import json
import os
import sqlite3
import tempfile

import pytest

from database import AnalysisDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield AnalysisDB(db_path=path)
    try:
        os.unlink(path)
    except OSError:
        pass


def _evento(db, device_id, losses=None, event_name='app_opened'):
    props = json.dumps({'_losses': losses}) if losses else None
    db.log_event(device_id=device_id, event_name=event_name, props=props,
                 platform='macos-dmg')


# ============================================================================
# LA MEDIDA
# ============================================================================

def test_sin_perdidas_el_reparto_esta_vacio(db):
    _evento(db, 'dev-1')
    r = db.telemetry_losses()
    assert r['by_cause'] == {}
    assert r['devices_reporting'] == 0
    assert r['devices_total'] == 1


def test_cuenta_las_dos_causas_POR_SEPARADO(db):
    _evento(db, 'dev-1', {'sin_red': 12, 'rechazado': 3})
    r = db.telemetry_losses()
    assert r['by_cause'] == {'sin_red': 12, 'rechazado': 3}


def test_MAX_por_dispositivo_no_suma(db):
    """El cliente manda un total acumulado desde la instalacion, adjunto a
    CADA evento que sale. Sumar las filas contaria el mismo evento perdido
    tantas veces como informes lo lleven — el mismo fallo que ya mordio con el
    consenso comunitario contando por marca de tiempo."""
    _evento(db, 'dev-1', {'sin_red': 5})
    _evento(db, 'dev-1', {'sin_red': 7})
    _evento(db, 'dev-1', {'sin_red': 9})
    assert db.telemetry_losses()['by_cause'] == {'sin_red': 9}


def test_entre_dispositivos_SI_suma(db):
    _evento(db, 'dev-1', {'sin_red': 4})
    _evento(db, 'dev-2', {'sin_red': 6})
    r = db.telemetry_losses()
    assert r['by_cause'] == {'sin_red': 10}
    assert r['devices_reporting'] == 2


def test_un_informe_que_se_cae_no_pierde_la_cuenta(db):
    """La razon de que sea monotonico: si el evento que llevaba el 5 no llega,
    el siguiente trae el 9 igual. Un delta «desde el ultimo envio» habria
    perdido esos cuatro para siempre."""
    _evento(db, 'dev-1', {'sin_red': 9})  # el del 5 nunca llego
    assert db.telemetry_losses()['by_cause'] == {'sin_red': 9}


def test_devices_reporting_es_el_denominador_honesto(db):
    """No es «todo el parque»: solo los aparatos que han mandado la clave. Con
    el total como denominador, la tasa de perdida saldria diluida por todos los
    que aun corren un cliente sin esto."""
    _evento(db, 'dev-1', {'sin_red': 2})
    _evento(db, 'dev-2')
    _evento(db, 'dev-3')
    r = db.telemetry_losses()
    assert r['devices_reporting'] == 1
    assert r['devices_total'] == 3


# ============================================================================
# LO QUE NO PUEDE ROMPER
# ============================================================================

def test_un_props_corrupto_no_tumba_el_agregado(db):
    conn = sqlite3.connect(db.db_path)
    try:
        conn.execute(
            "INSERT INTO events (device_id, event_name, props) "
            "VALUES ('roto', 'app_opened', '{\"_losses\": no-es-json')")
        conn.commit()
    finally:
        conn.close()
    _evento(db, 'dev-1', {'sin_red': 3})
    assert db.telemetry_losses()['by_cause'] == {'sin_red': 3}


def test_un_valor_que_no_es_numero_se_ignora(db):
    """`losses` lo manda el CLIENTE y va derecho a un agregado."""
    _evento(db, 'dev-1', {'sin_red': 'muchos', 'rechazado': 2})
    assert db.telemetry_losses()['by_cause'] == {'rechazado': 2}


def test_los_eventos_normales_no_se_ven_afectados(db):
    """La medida viaja dentro de `props`, bajo una clave reservada, para no
    tocar el esquema de `events` ni la purga."""
    db.log_event(device_id='dev-1', event_name='import_completed',
                 props=json.dumps({'count': 42}), platform='windows')
    conn = sqlite3.connect(db.db_path)
    try:
        fila = conn.execute(
            "SELECT props FROM events WHERE event_name='import_completed'"
        ).fetchone()
    finally:
        conn.close()
    assert json.loads(fila[0]) == {'count': 42}
    assert db.telemetry_losses()['by_cause'] == {}


def test_la_ventana_deja_fuera_lo_viejo(db):
    conn = sqlite3.connect(db.db_path)
    try:
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name, props) "
            "VALUES (datetime('now','-45 days'), 'viejo', 'app_opened', ?)",
            (json.dumps({'_losses': {'sin_red': 999}}),))
        conn.commit()
    finally:
        conn.close()
    _evento(db, 'dev-1', {'sin_red': 2})
    assert db.telemetry_losses(days=30)['by_cause'] == {'sin_red': 2}


def test_dice_lo_que_NO_puede_ver(db):
    """Un aparato que pierde eventos y no vuelve a conectar jamas no aparece
    aqui, y ese suelo es irreducible. Escrito en la respuesta para que nadie
    lea el numero como «las perdidas totales»."""
    r = db.telemetry_losses()
    assert 'note' in r
    assert 'irreducible' in r['note']


# ============================================================================
# EL CABLEADO
#
# Por HTTP de verdad, no leyendo el fuente. En este proyecto los bugs de
# wiring han sido los mas caros —`admin_sync_router` llevaba desde siempre
# definido y sin `include_router`, cinco rutas dando 404 en produccion— y un
# test que busca una cadena en un fichero no habria visto ninguno de ellos.
# ============================================================================

class TestPorHTTP:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_client_event_acepta_losses_y_las_SANEA(self, client):
        """`losses` lo manda el cliente y va derecho a un agregado, asi que
        entra por el mismo sitio que cualquier otro dato de fuera."""
        import main

        dev = 'test_losses_%d' % os.getpid()
        r = client.post(
            '/client-event',
            json={
                'event_name': 'app_opened',
                'props': {'count': 7},
                'losses': {'sin_red': 4, 'rechazado': 'muchos', 'raro': -1},
            },
            headers={'X-Device-Id': dev},
        )
        assert r.status_code == 202
        assert r.json()['logged'] is True

        conn = sqlite3.connect(main.db.db_path)
        try:
            fila = conn.execute(
                "SELECT props FROM events WHERE device_id = ?", (dev,)
            ).fetchone()
        finally:
            conn.close()
        assert fila is not None
        props = json.loads(fila[0])
        # Lo que el cliente mandaba de siempre sigue intacto...
        assert props['count'] == 7
        # ...y las perdidas viajan aparte, bajo la clave reservada, sin la
        # basura: 'muchos' no es un numero y -1 no es una perdida.
        assert props['_losses'] == {'sin_red': 4}

    def test_un_evento_sin_losses_no_crece_ni_un_byte(self, client):
        """La mayoria de los aparatos no ha perdido un evento en su vida."""
        import main

        dev = 'test_sin_losses_%d' % os.getpid()
        r = client.post(
            '/client-event',
            json={'event_name': 'app_opened', 'props': {'count': 1}},
            headers={'X-Device-Id': dev},
        )
        assert r.status_code == 202
        conn = sqlite3.connect(main.db.db_path)
        try:
            fila = conn.execute(
                "SELECT props FROM events WHERE device_id = ?", (dev,)
            ).fetchone()
        finally:
            conn.close()
        assert json.loads(fila[0]) == {'count': 1}

    def test_admin_telemetry_lo_ENSEÑA(self, client, monkeypatch):
        """El dato existiendo y no enseñandose es el patron que mas veces ha
        mordido esta semana. Aqui se pide el endpoint de verdad."""
        import routes.admin_panel as ap

        # El panel lee de DOS bases distintas y esta corre sin `sync.db`. Una
        # vacia con el esquema minimo basta: lo que se comprueba aqui es que la
        # clave sale por HTTP, no lo que valga.
        fd, sync_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        conn = sqlite3.connect(sync_path)
        conn.execute(
            'CREATE TABLE sync_items ('
            ' id INTEGER PRIMARY KEY AUTOINCREMENT, last_device_id TEXT,'
            ' device_type TEXT, data_type TEXT,'
            " item_key TEXT NOT NULL DEFAULT '', payload TEXT)")
        conn.commit()
        conn.close()
        monkeypatch.setattr(ap, '_SYNC_DB_PATH', sync_path)

        secreto = 'test-admin-secret-1234567890'
        monkeypatch.setenv('ADMIN_TOKEN', secreto)
        try:
            r = client.get('/admin/telemetry',
                           headers={'X-Admin-Secret': secreto})
            assert r.status_code == 200
            cuerpo = r.json()
            assert 'telemetry_losses' in cuerpo
            # Y con la forma completa, no un `{}` del except que se lo traga.
            assert 'by_cause' in cuerpo['telemetry_losses']
            assert 'devices_reporting' in cuerpo['telemetry_losses']
        finally:
            try:
                os.unlink(sync_path)
            except OSError:
                pass
