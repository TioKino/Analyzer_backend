"""SEC-12 (b): el consenso de la comunidad cuenta por CUENTA, no por aparato.

Hasta el 2026-09-26 contaba por aparato, y eso tenía dos caras:

  - El DJ con escritorio + móvil + tablet valía TRES votos siendo una opinión.
    Un `consensus_3` (prioridad 80, gana al motor local y al id3) podía salir
    de una sola persona.
  - Los cues viajan por sync entre los aparatos de la misma persona. Si los
    dos los suben a /community/cues, una sola persona fabricaba una zona
    «de 2 DJs» consigo misma.

La cuenta sale de sync.db (`user_devices`) y se resuelve al LEER: vincular un
aparato después de votar tiene que juntar sus votos, cosa que un `user_id`
guardado al escribir no haría (vincular MUEVE el aparato a la otra cuenta).

    pytest test_consenso_por_cuenta.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from community_cues_endpoint import aggregate_cues_into_zones, contribuyentes  # noqa: E402
from database import AnalysisDB  # noqa: E402

FP = 'a' * 32


@pytest.fixture
def db(tmp_path):
    d = AnalysisDB(db_path=str(tmp_path / 'analysis.db'))
    d.cuentas = {}
    d.cuentas_de = lambda ids: {i: d.cuentas[i] for i in ids if i in d.cuentas}
    return d


def _corrige(db, device_id, valor, cuando):
    db.save_correction('t1', 'genre', None, valor, FP, device_id)
    conn = db._open_conn()
    conn.execute('UPDATE corrections SET corrected_at = ? WHERE device_id = ? '
                 'AND fingerprint = ? AND field = ?', (cuando, device_id, FP, 'genre'))
    conn.commit()
    conn.close()


class TestCorrecciones:
    def test_tres_aparatos_de_una_persona_son_UN_voto(self, db):
        db.cuentas = {'mac': 'u1', 'iphone': 'u1', 'ipad': 'u1'}
        for i, d in enumerate(('mac', 'iphone', 'ipad')):
            _corrige(db, d, 'Techno', f'2026-09-26T10:0{i}:00')
        assert db.get_consensus(FP, 'genre') == ('Techno', 1)
        assert db.get_consensus(FP, 'genre', min_votes=3) == (None, 0), (
            'una persona sola no fabrica un consensus_3')
        assert db.get_all_consensus(FP)['genre'] == ('Techno', 1)

    def test_tres_personas_SI_son_tres(self, db):
        db.cuentas = {'mac': 'u1', 'pc': 'u2'}  # el tercero no se ha registrado
        for i, d in enumerate(('mac', 'pc', 'suelto')):
            _corrige(db, d, 'Techno', f'2026-09-26T10:0{i}:00')
        assert db.get_consensus(FP, 'genre', min_votes=3) == ('Techno', 3)

    def test_la_misma_persona_cambiando_de_opinion_vale_lo_ultimo(self, db):
        db.cuentas = {'mac': 'u1', 'iphone': 'u1'}
        # El viejo va del aparato que SQLite devuelve primero (índice por
        # device_id): si ganara el primero que sale, ganaría el viejo.
        _corrige(db, 'iphone', 'House', '2026-09-26T10:00:00')
        _corrige(db, 'mac', 'Techno', '2026-09-26T11:00:00')
        assert db.get_consensus(FP, 'genre') == ('Techno', 1)
        assert db.get_all_consensus(FP)['genre'] == ('Techno', 1)

    def test_vincular_DESPUES_de_votar_junta_los_votos(self, db):
        _corrige(db, 'mac', 'Techno', '2026-09-26T10:00:00')
        _corrige(db, 'iphone', 'Techno', '2026-09-26T10:01:00')
        assert db.get_consensus(FP, 'genre') == ('Techno', 2)
        db.cuentas = {'mac': 'u1', 'iphone': 'u1'}
        assert db.get_consensus(FP, 'genre') == ('Techno', 1)

    def test_las_filas_viejas_sin_aparato_siguen_contando_una_a_una(self, db):
        conn = db._open_conn()
        for i in range(3):
            conn.execute(
                'INSERT INTO corrections (track_id, field, new_value, corrected_at, '
                'fingerprint) VALUES (?,?,?,?,?)',
                ('t1', 'genre', 'Techno', f'2025-01-0{i + 1}', FP))
        conn.commit()
        conn.close()
        assert db.get_consensus(FP, 'genre', min_votes=3) == ('Techno', 3)

    def test_sin_sync_db_cuenta_por_aparato_como_antes(self, db):
        def roto(ids):
            raise RuntimeError('sync.db no responde')
        db.cuentas_de = roto
        _corrige(db, 'mac', 'Techno', '2026-09-26T10:00:00')
        _corrige(db, 'iphone', 'Techno', '2026-09-26T10:01:00')
        assert db.get_consensus(FP, 'genre') == ('Techno', 2)


class TestVotosDeLaComunidad:
    def _vota(self, db, device_id, field, valor):
        db.submit_community_override(FP, device_id, field, valor)

    def test_tres_aparatos_de_una_persona_no_ganan(self, db):
        db.cuentas = {'mac': 'u1', 'iphone': 'u1', 'ipad': 'u1'}
        for d in ('mac', 'iphone', 'ipad'):
            self._vota(db, d, 'track_type', 'peak_time')
        assert db.get_community_votes(FP, 'track_type') == {'peak_time': 1}
        assert db.get_community_consensus(FP, 'track_type') is None

    def test_tres_personas_ganan(self, db):
        db.cuentas = {'mac': 'u1', 'pc': 'u2', 'iphone': 'u3'}
        for d in ('mac', 'pc', 'iphone'):
            self._vota(db, d, 'track_type', 'peak_time')
        c = db.get_community_consensus(FP, 'track_type')
        assert c['value'] == 'peak_time' and c['votes'] == 3 and c['total'] == 3

    def test_el_numerico_cuenta_igual(self, db):
        db.cuentas = {'mac': 'u1', 'iphone': 'u1', 'ipad': 'u1'}
        for d in ('mac', 'iphone', 'ipad'):
            self._vota(db, d, 'bpm', '128.0')
        r = db.get_community_consensus_numeric(FP, 'bpm')
        assert r['total_voters'] == 1 and r['consensus'] is None


class TestZonasDeCues:
    def _cues(self, *devices):
        return [{'cue_type': 'drop', 'device_id': d, 'position_ms': 60000 + i * 100,
                 'end_position_ms': None, 'note': None}
                for i, d in enumerate(devices)]

    def test_los_cues_de_una_persona_en_dos_aparatos_no_son_una_zona(self):
        filas = self._cues('mac', 'iphone')
        assert len(aggregate_cues_into_zones(filas)) == 1, 'contando aparatos salía zona'
        cuentas = {'mac': 'u1', 'iphone': 'u1'}
        assert aggregate_cues_into_zones(filas, 0, cuentas) == []
        assert contribuyentes(filas, cuentas) == 1

    def test_dos_personas_si_hacen_zona(self):
        filas = self._cues('mac', 'pc')
        zonas = aggregate_cues_into_zones(filas, 0, {'mac': 'u1', 'pc': 'u2'})
        assert len(zonas) == 1 and zonas[0]['dj_count'] == 2


def test_main_cablea_las_cuentas():
    import main

    assert main.db.cuentas_de is not None, (
        'sin el resolver cada aparato vuelve a contar como un DJ')


def test_el_resolver_lee_user_devices(tmp_path, monkeypatch):
    import sync_endpoints as se

    monkeypatch.setattr(se, '_DB_PATH', str(tmp_path / 'sync.db'))
    monkeypatch.setattr(se, '_conn', None)
    conn = se._get_conn()
    conn.execute("INSERT INTO users (user_id, created_at) VALUES ('u1', ?)", (se._now_iso(),))
    for d in ('mac', 'iphone'):
        conn.execute(
            "INSERT INTO user_devices (device_id, user_id, device_type, device_name, "
            "linked_at) VALUES (?, 'u1', 'x', 'x', ?)", (d, se._now_iso()))
    conn.commit()
    try:
        assert se.cuentas_de_dispositivos(['mac', 'iphone', 'suelto', None]) == {
            'mac': 'u1', 'iphone': 'u1'}
    finally:
        conn.close()
        monkeypatch.setattr(se, '_conn', None)
