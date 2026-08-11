"""
Cobertura de huella acústica en el panel admin.

POR QUÉ EXISTE (2026-08-10): el backfill de huella corre SOLO y en background
en el cliente desktop (2 s/track, resumible), y nadie tenía forma de saber
cuánto llevaba hecho. El panel medía preview y artwork pero no la huella, así
que la pregunta "¿está la memoria colectiva por sonido realmente viva?" no se
podía responder con datos — sólo abriendo la BD por SQL a mano.

LA MÉTRICA QUE IMPORTA es `tracks_in_multi_clusters`, no `with_chromaprint`.
Un track con huella que está solo en su cluster no comparte NADA con nadie: la
promesa del producto (cues, beat-grid y correcciones compartidos entre copias
distintas del mismo audio) sólo se cumple cuando hay dos o más agrupados. Es
perfectamente posible tener el 100 % de huellas calculadas y cero memoria
compartida, y el panel debe dejar ver esa diferencia.
"""

import os
import tempfile
import uuid

import pytest


@pytest.fixture()
def db():
    from database import AnalysisDB

    return AnalysisDB(tempfile.mktemp(suffix=".db"))


def _track(db, *, chromaprint=None, acoustic_id=None):
    """Inserta un track directo por SQL: aquí se prueba el CONTEO, no el
    pipeline de análisis."""
    fp = uuid.uuid4().hex
    conn = db._open_conn()
    try:
        conn.execute(
            "INSERT INTO tracks (fingerprint, filename, artist, title, bpm, "
            "duration, chromaprint, acoustic_id) VALUES (?,?,?,?,?,?,?,?)",
            (fp, f"{fp}.mp3", "Artista", "Tema", 128.0, 300.0,
             chromaprint, acoustic_id),
        )
        conn.commit()
    finally:
        conn.close()
    return fp


class TestCobertura:
    def test_bd_vacia_no_divide_por_cero(self, db):
        c = db.acoustic_coverage()
        assert c['total_tracks'] == 0
        assert c['chromaprint_pct'] is None, (
            "con 0 tracks el porcentaje debe ser None, no 0.0: un 0 % se lee "
            "como 'el backfill no ha hecho nada' cuando no hay nada que hacer"
        )

    def test_cuenta_los_que_tienen_huella(self, db):
        _track(db, chromaprint="AQAA")
        _track(db, chromaprint="AQAB")
        _track(db)  # sin huella
        c = db.acoustic_coverage()
        assert c['total_tracks'] == 3
        assert c['with_chromaprint'] == 2
        assert c['without_chromaprint'] == 1
        assert c['chromaprint_pct'] == pytest.approx(66.7, abs=0.1)

    def test_la_cadena_vacia_no_cuenta_como_huella(self, db):
        """`chromaprint = ''` es "se intentó y no salió", no "tiene huella".
        Contarlo inflaría la cobertura y daría el backfill por terminado."""
        _track(db, chromaprint="")
        c = db.acoustic_coverage()
        assert c['with_chromaprint'] == 0
        assert c['without_chromaprint'] == 1


class TestLoQueDeVerdadImporta:
    def test_huella_en_solitario_no_es_memoria_compartida(self, db):
        """El caso que engaña: 3 tracks con huella, cada uno en su cluster.
        Cobertura del 100 % y CERO memoria compartida."""
        for i in range(3):
            _track(db, chromaprint=f"AQ{i}", acoustic_id=f"cluster-{i}")
        c = db.acoustic_coverage()
        assert c['chromaprint_pct'] == 100.0
        assert c['with_cluster'] == 3
        assert c['multi_clusters'] == 0
        assert c['tracks_in_multi_clusters'] == 0, (
            "tres clusters de uno no comparten nada; contarlos como memoria "
            "compartida haría creer que la promesa se cumple cuando no"
        )

    def test_dos_copias_del_mismo_audio_si_cuentan(self, db):
        _track(db, chromaprint="AQAA", acoustic_id="mismo-sonido")
        _track(db, chromaprint="AQAB", acoustic_id="mismo-sonido")
        c = db.acoustic_coverage()
        assert c['multi_clusters'] == 1
        assert c['tracks_in_multi_clusters'] == 2

    def test_mezcla_realista(self, db):
        """Lo que se espera de una biblioteca de verdad: un grupo de 3 copias,
        un par, varios solitarios y algunos sin huella todavía."""
        for _ in range(3):
            _track(db, chromaprint="X", acoustic_id="triple")
        for _ in range(2):
            _track(db, chromaprint="Y", acoustic_id="pareja")
        for i in range(4):
            _track(db, chromaprint="Z", acoustic_id=f"solo-{i}")
        for _ in range(5):
            _track(db)  # aún sin backfillear

        c = db.acoustic_coverage()
        assert c['total_tracks'] == 14
        assert c['with_chromaprint'] == 9
        assert c['with_cluster'] == 9
        assert c['multi_clusters'] == 2          # triple + pareja
        assert c['tracks_in_multi_clusters'] == 5  # 3 + 2
        assert c['chromaprint_pct'] == pytest.approx(64.3, abs=0.1)

    def test_los_sin_cluster_no_entran_en_el_grupo(self, db):
        """`acoustic_id IS NULL` no puede agruparse con otro NULL: son tracks
        sin huella, no un cluster gigante de desconocidos."""
        for _ in range(5):
            _track(db, chromaprint="W")  # huella pero sin cluster asignado
        c = db.acoustic_coverage()
        assert c['multi_clusters'] == 0
        assert c['tracks_in_multi_clusters'] == 0


class TestPanelAdmin:
    def test_el_telemetry_lo_expone(self):
        from routes.admin_panel import _acoustic_coverage

        c = _acoustic_coverage()
        assert isinstance(c, dict)

    def test_no_tumba_el_panel_si_la_bd_falla(self, monkeypatch):
        """Una métrica de observación jamás debe dejar al owner sin panel."""
        import routes.admin_panel as ap

        class BDRota:
            def acoustic_coverage(self):
                raise RuntimeError("BD caída")

        monkeypatch.setattr(ap, "_get_db", lambda: BDRota())
        assert ap._acoustic_coverage() == {}


class TestRadioDeImpactoDeSEC01:
    """`rows_without_md5_fingerprint` mide cuantas filas dejaron de poder dar
    cache-hit tras SEC-01. Es un numero operativo: decide si el arreglo provoca
    una tormenta de reanalisis en un Render de un solo worker o si es inocuo.

    OJO con los dos "fingerprint" del proyecto: `chromaprint` es la huella
    ACUSTICA (agrupa por sonido) y `fingerprint` es el MD5 del CONTENIDO. Este
    contador va del segundo."""

    def test_cuenta_las_filas_sin_md5(self, db):
        _track(db, chromaprint="AQAA")           # con MD5 (lo pone _track)
        conn = db._open_conn()
        try:
            conn.execute(
                "INSERT INTO tracks (fingerprint, filename, bpm, duration) "
                "VALUES ('', 'sin_md5.mp3', 128.0, 300.0)")
            conn.execute(
                "INSERT INTO tracks (fingerprint, filename, bpm, duration) "
                "VALUES (NULL, 'null_md5.mp3', 128.0, 300.0)")
            conn.commit()
        finally:
            conn.close()

        c = db.acoustic_coverage()
        assert c['rows_without_md5_fingerprint'] == 2, (
            "la cadena vacia y el NULL cuentan igual: ninguna de las dos "
            "permite verificar que el track sea tuyo"
        )

    def test_no_se_confunde_con_la_huella_acustica(self, db):
        """Una fila puede tener MD5 y no tener chromaprint — es de hecho el
        caso mayoritario hoy (28,5% de cobertura acustica). Contarlas aqui
        inflaria el radio de impacto y daria una falsa alarma."""
        for _ in range(5):
            _track(db)  # MD5 si, chromaprint no
        c = db.acoustic_coverage()
        assert c['without_chromaprint'] == 5
        assert c['rows_without_md5_fingerprint'] == 0, (
            "se estan contando filas sin huella ACUSTICA como si les faltara "
            "el MD5: son cosas distintas"
        )
