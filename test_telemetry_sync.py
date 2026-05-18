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
            payload TEXT
        )
    ''')
    return conn, path


def _insert_sync_item(conn, *, device_id, device_type, tracks):
    """Inserta una fila de sync_items con data_type='analysis'."""
    payload = json.dumps({'tracks': tracks})
    conn.execute(
        'INSERT INTO sync_items (last_device_id, device_type, data_type, payload) '
        "VALUES (?, ?, 'analysis', ?)",
        (device_id, device_type, payload),
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

    def test_dos_devices_mismo_fingerprint_es_colision(self, empty_sync):
        _insert_sync_item(empty_sync,
            device_id='dja_a', device_type='macos',
            tracks={'t1': {'fingerprint': 'SHARED'}},
        )
        _insert_sync_item(empty_sync,
            device_id='dja_b', device_type='ios',
            tracks={'t2': {'fingerprint': 'SHARED'}},
        )
        out = _compute_telemetry_from_sync(empty_sync)
        assert out['fingerprints']['total_tracks'] == 2
        assert out['fingerprints']['unique_fingerprints'] == 1
        assert out['fingerprints']['collision_groups'] == 1
        assert out['fingerprints']['collision_extra_rows'] == 1
        assert out['total_users'] == 2
        assert out['platforms'] == {'macos': 1, 'ios': 1}

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
