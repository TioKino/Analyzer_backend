"""Tests para _compute_telemetry_from_sync: la funcion que deriva
fingerprints + sources + users desde sync_items (sync.db) en lugar
de analysis.db.
"""

import json
import os
import sqlite3
import tempfile

import pytest

from routes.admin_panel import _compute_telemetry_from_sync


def _make_sync_db():
    """Crea sync.db temporal con schema minimo de sync_items."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE sync_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_device_id TEXT,
            device_type TEXT,
            data_type TEXT,
            item_key TEXT NOT NULL DEFAULT '',
            payload TEXT
        )
    ''')
    return conn, path


def _insert_sync_item(conn, *, device_id, device_type, tracks):
    """Inserta un blob legacy (clientes < v2.9.3): item_key='all_analysis',
    payload = {"tracks": {trackId: {...}}}."""
    payload = json.dumps({'tracks': tracks})
    conn.execute(
        'INSERT INTO sync_items '
        '(last_device_id, device_type, data_type, item_key, payload) '
        "VALUES (?, ?, 'analysis', 'all_analysis', ?)",
        (device_id, device_type, payload),
    )
    conn.commit()


def _insert_incremental(conn, *, device_id, device_type, track_id, track):
    """Inserta una fila del formato incremental (v2.9.3+): una fila = UN track,
    item_key=<trackId>, payload = el track suelto SIN envolver en {"tracks"}.

    Este es el formato que usan los clientes actuales y el que el panel no
    sabia leer: _parse_analysis_payload devolvia el track entero y se iteraban
    sus CAMPOS como si cada uno fuera un track."""
    conn.execute(
        'INSERT INTO sync_items '
        '(last_device_id, device_type, data_type, item_key, payload) '
        "VALUES (?, ?, 'analysis', ?, ?)",
        (device_id, device_type, track_id, json.dumps(track)),
    )
    conn.commit()


@pytest.fixture
def empty_sync():
    conn, path = _make_sync_db()
    try:
        yield conn
    finally:
        conn.close()
        try:
            os.unlink(path)
        except OSError:
            pass


class TestComputeTelemetryFromSync:

    def test_empty_devuelve_zeros(self, empty_sync):
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 0
        assert out['sources']['bpm'] == {}
        assert out['total_users'] == 0
        assert out['platforms'] == {}

    def test_un_device_dos_tracks_camelCase(self, empty_sync):
        _insert_sync_item(empty_sync,
            device_id='dja_abc', device_type='macos',
            tracks={
                't1': {'bpmSource': 'rekordbox', 'fingerprint': 'fp1'},
                't2': {'bpmSource': 'local_engine', 'fingerprint': 'fp2'},
            },
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 2
        assert out['fingerprints']['with_fingerprint'] == 2
        assert out['sources']['bpm']['rekordbox'] == 1
        assert out['sources']['bpm']['local_engine'] == 1
        assert out['total_users'] == 1
        assert out['platforms'] == {'macos': 1}

    def test_snake_case_tambien_funciona(self, empty_sync):
        _insert_sync_item(empty_sync,
            device_id='dja_xyz', device_type='windows',
            tracks={
                't1': {'bpm_source': 'traktor', 'fingerprint': 'fpx'},
            },
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['sources']['bpm']['traktor'] == 1
        assert out['platforms'] == {'windows': 1}

    def test_mismo_track_en_dos_devices_cuenta_una_vez(self, empty_sync):
        """El caso normal: el usuario tiene el mismo fichero en PC y movil.

        Antes se contaba 2 veces y ademas salia como 'colision de fingerprint',
        que es justo lo contrario de lo que esa metrica quiere decir."""
        _insert_sync_item(empty_sync,
            device_id='dja_a', device_type='macos',
            tracks={'t1': {'id': 'SAME', 'fingerprint': 'SHARED'}},
        )
        _insert_sync_item(empty_sync,
            device_id='dja_b', device_type='ios',
            tracks={'t1': {'id': 'SAME', 'fingerprint': 'SHARED'}},
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['fingerprints']['unique_fingerprints'] == 1
        assert out['fingerprints']['collision_groups'] == 0
        assert out['fingerprints']['collision_extra_rows'] == 0
        # Los devices siguen contando por separado: son 2 dispositivos reales.
        assert out['total_users'] == 2
        assert out['platforms'] == {'macos': 1, 'ios': 1}

    def test_dos_tracks_distintos_mismo_fingerprint_si_es_colision(self, empty_sync):
        """Colision de verdad: identidades distintas (otro nombre/tamano de
        fichero) que comparten el mismo audio. Es lo que la memoria colectiva
        por sonido puede unificar."""
        _insert_sync_item(empty_sync,
            device_id='dja_a', device_type='macos',
            tracks={
                't1': {'id': 'ID_A', 'fingerprint': 'SHARED'},
                't2': {'id': 'ID_B', 'fingerprint': 'SHARED'},
            },
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 2
        assert out['fingerprints']['unique_fingerprints'] == 1
        assert out['fingerprints']['collision_groups'] == 1
        assert out['fingerprints']['collision_extra_rows'] == 1

    def test_track_sin_source_cae_en_unknown(self, empty_sync):
        _insert_sync_item(empty_sync,
            device_id='dja_a', device_type='ios',
            tracks={'t1': {'fingerprint': 'fp'}},  # sin bpmSource
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['sources']['bpm']['unknown'] == 1

    def test_payload_invalido_no_explota(self, empty_sync):
        empty_sync.execute(
            "INSERT INTO sync_items (last_device_id, device_type, data_type, payload) "
            "VALUES ('dja_x', 'ios', 'analysis', 'NOT JSON')"
        )
        empty_sync.commit()
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 0
        # device sigue contando (lo vimos en sync_items aunque el payload este roto)
        assert out['total_users'] == 1

    def test_platforms_lowercase(self, empty_sync):
        # El cliente envia "macos" / "iOS" / "Windows" — normalizamos
        _insert_sync_item(empty_sync,
            device_id='dja_a', device_type='MacOS',
            tracks={'t1': {'fingerprint': 'fp'}},
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert 'macos' in out['platforms']


class TestFormatoIncremental:
    """Regresion del bug que inflaba el panel a 304.232 'tracks'.

    Los clientes v2.9.3+ suben el analisis track a track (una fila de
    sync_items por track, item_key=<trackId>, payload = el track suelto).
    _compute_telemetry_from_sync leia solo `payload`, sin item_key, asi que no
    podia distinguir ese formato del blob legacy: pasaba el track por
    _parse_analysis_payload (que devuelve el propio dict al no encontrar la
    clave "tracks") e iteraba sus CAMPOS como si cada uno fuera un track.

    Doble efecto: se inventaba miles de tracks fantasma con todas las fuentes
    en 'unknown', y los tracks REALES de los clientes actuales no se contaban
    nunca — solo aportaban datos los clientes legacy.
    """

    def test_fila_incremental_cuenta_un_track(self, empty_sync):
        _insert_incremental(empty_sync,
            device_id='dja_a', device_type='macos', track_id='trk1',
            track={
                'id': 'trk1',
                'fingerprint': 'fp1',
                'bpmSource': 'rekordbox',
                'keySource': 'rekordbox',
            },
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['fingerprints']['with_fingerprint'] == 1
        # Y las fuentes se leen del track, no se pierden.
        assert out['sources']['bpm']['rekordbox'] == 1
        assert out['sources']['key']['rekordbox'] == 1

    def test_campos_del_track_no_se_cuentan_como_tracks(self, empty_sync):
        """El track lleva campos anidados (dicts y listas). Antes cada campo
        que fuera dict sumaba 1 al contador y caia en 'unknown'."""
        _insert_incremental(empty_sync,
            device_id='dja_a', device_type='macos', track_id='trk1',
            track={
                'id': 'trk1',
                'fingerprint': 'fp1',
                'bpmSource': 'analysis',
                'structure': {'intro': 8, 'outro': 16},
                'characteristics': {'heavy_bass': True, 'has_drop': True},
                'cuePoints': [{'pos': 1.0}, {'pos': 2.0}],
                'energyCurve': [1, 2, 3],
            },
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['sources']['bpm'] == {'analysis': 1}
        assert 'unknown' not in out['sources']['bpm']

    def test_incremental_dedupe_entre_devices(self, empty_sync):
        """Mismo track subido desde PC y desde movil = 1 track, 2 devices."""
        for dev, plat in (('dja_pc', 'macos'), ('dja_phone', 'ios')):
            _insert_incremental(empty_sync,
                device_id=dev, device_type=plat, track_id='trk1',
                track={'id': 'trk1', 'fingerprint': 'fp1', 'bpmSource': 'analysis'},
            )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['sources']['bpm'] == {'analysis': 1}
        assert out['total_users'] == 2

    def test_dedupe_entre_formato_legacy_e_incremental(self, empty_sync):
        """El mismo track en un cliente viejo (blob) y uno nuevo (incremental)
        es UN track. Antes ni siquiera compartian criterio de identidad: la
        rama incremental miraba fingerprint primero y la legacy id primero."""
        _insert_sync_item(empty_sync,
            device_id='dja_old', device_type='windows',
            tracks={'trk1': {'id': 'trk1', 'fingerprint': 'fp1'}},
        )
        _insert_incremental(empty_sync,
            device_id='dja_new', device_type='macos', track_id='trk1',
            track={'id': 'trk1', 'fingerprint': 'fp1'},
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['fingerprints']['collision_groups'] == 0

    def test_gana_el_payload_mas_completo(self, empty_sync):
        """El movil manda menos campos que el PC. Al deduplicar nos quedamos
        con el mas rico para no perder bpm_source / artwork segun el orden en
        que se lean las filas."""
        _insert_incremental(empty_sync,
            device_id='dja_phone', device_type='ios', track_id='trk1',
            track={'id': 'trk1'},
        )
        _insert_incremental(empty_sync,
            device_id='dja_pc', device_type='macos', track_id='trk1',
            track={
                'id': 'trk1',
                'fingerprint': 'fp1',
                'bpmSource': 'rekordbox',
                'artworkUrl': 'https://x/y.jpg',
            },
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['fingerprints']['with_fingerprint'] == 1
        assert out['sources']['bpm'] == {'rekordbox': 1}
        assert out['artwork_coverage']['with_artwork'] == 1

    def test_item_key_sirve_de_identidad_si_falta_id(self, empty_sync):
        """Payload sin id ni fingerprint: el item_key ES el trackId."""
        _insert_incremental(empty_sync,
            device_id='dja_a', device_type='macos', track_id='trk_sin_id',
            track={'bpmSource': 'id3'},
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 1
        assert out['fingerprints']['with_fingerprint'] == 0
        assert out['sources']['bpm'] == {'id3': 1}

    def test_incremental_con_payload_roto_no_explota(self, empty_sync):
        empty_sync.execute(
            "INSERT INTO sync_items "
            "(last_device_id, device_type, data_type, item_key, payload) "
            "VALUES ('dja_x', 'ios', 'analysis', 'trk1', 'NOT JSON')"
        )
        empty_sync.commit()
        out = _compute_telemetry_from_sync(empty_sync)
        # La fila existe y su item_key identifica un track, aunque el payload
        # este roto: cuenta como track sin datos, no revienta ni se inventa 10.
        assert out['fingerprints']['total_tracks'] == 1
        assert out['fingerprints']['with_fingerprint'] == 0
        assert out['total_users'] == 1
