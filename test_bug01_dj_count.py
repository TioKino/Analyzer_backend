"""
BUG-01 — `dj_count` contaba DJs distintos... y siempre valía 1.

`increment_popularity()` recibía `device_id` y **lo ignoraba**: el INSERT ponía
`dj_count = 1` y el UPDATE solo tocaba `analysis_count`. Resultado:
`/community/popularity/{fingerprint}` devolvía 1 para todos los tracks, para
siempre. No es un crash: es un dato que la API afirma con confianza y es falso.

Lo que hace falta para contarlo bien es saber QUIÉNES son (`track_analyzers`),
porque un contador ciego no distingue "diez DJs distintos" de "un DJ que
reanalizó diez veces" — y esa diferencia es justo lo que el número promete.
"""

import tempfile
import uuid

import pytest


@pytest.fixture()
def db():
    from database import AnalysisDB

    return AnalysisDB(tempfile.mktemp(suffix=".db"))


def _pop(db, fp):
    return db.get_track_popularity(fp) if hasattr(db, 'get_track_popularity') \
        else db.get_popularity(fp)


class TestCuentaDJsDistintos:
    def test_tres_djs_distintos_cuentan_tres(self, db):
        fp = uuid.uuid4().hex
        for dev in ("dj-a", "dj-b", "dj-c"):
            db.increment_popularity(fp, dev)
        assert _pop(db, fp)['dj_count'] == 3

    def test_el_mismo_dj_diez_veces_sigue_siendo_uno(self, db):
        """EL CORAZÓN DEL DATO. Si esto contara diez, el número dejaría de
        significar 'cuánta gente usa este track' y pasaría a ser un contador de
        reproducciones disfrazado."""
        fp = uuid.uuid4().hex
        for _ in range(10):
            db.increment_popularity(fp, "dj-obsesivo")
        p = _pop(db, fp)
        assert p['dj_count'] == 1, "un solo DJ se contó como varios"
        assert p['analysis_count'] == 10, (
            "analysis_count SÍ debe subir con cada análisis: son cosas distintas"
        )

    def test_mezcla_realista(self, db):
        fp = uuid.uuid4().hex
        for dev, veces in (("dj-a", 3), ("dj-b", 1), ("dj-c", 5)):
            for _ in range(veces):
                db.increment_popularity(fp, dev)
        p = _pop(db, fp)
        assert p['dj_count'] == 3
        assert p['analysis_count'] == 9

    def test_tracks_distintos_no_se_mezclan(self, db):
        fp_a, fp_b = uuid.uuid4().hex, uuid.uuid4().hex
        for dev in ("dj-1", "dj-2", "dj-3"):
            db.increment_popularity(fp_a, dev)
        db.increment_popularity(fp_b, "dj-1")
        assert _pop(db, fp_a)['dj_count'] == 3
        assert _pop(db, fp_b)['dj_count'] == 1


class TestElSueloDeUno:
    def test_sin_device_id_no_se_queda_en_cero(self, db):
        """Un cliente que no manda la cabecera no es identificable, pero el
        análisis ocurrió. Devolver 0 DJs sería más falso que devolver 1."""
        fp = uuid.uuid4().hex
        db.increment_popularity(fp, "")
        assert _pop(db, fp)['dj_count'] == 1

    def test_varios_anonimos_no_inflan_la_cuenta(self, db):
        """No se puede distinguir a un anónimo de otro, así que contarlos
        inventaría DJs que no sabemos que existan."""
        fp = uuid.uuid4().hex
        for _ in range(7):
            db.increment_popularity(fp, "")
        assert _pop(db, fp)['dj_count'] == 1

    def test_un_anonimo_no_borra_a_los_identificados(self, db):
        fp = uuid.uuid4().hex
        db.increment_popularity(fp, "dj-a")
        db.increment_popularity(fp, "dj-b")
        db.increment_popularity(fp, "")     # anónimo después
        assert _pop(db, fp)['dj_count'] == 2

    def test_las_filas_historicas_no_caen_a_cero(self, db):
        """Las filas anteriores a `track_analyzers` no tienen a nadie
        registrado. Recalcular a pelo las dejaría en 0 DJs, que es PEOR que el
        bug original: pasaríamos de 'siempre 1' a 'nadie lo usa'."""
        fp = uuid.uuid4().hex
        conn = db._open_conn()
        try:
            conn.execute(
                'INSERT INTO track_popularity '
                '(fingerprint, analysis_count, dj_count, last_analyzed) '
                'VALUES (?, 42, 1, ?)', (fp, '2026-01-01T00:00:00'),
            )
            conn.commit()
        finally:
            conn.close()

        db.increment_popularity(fp, "")
        p = _pop(db, fp)
        assert p['dj_count'] >= 1, "una fila histórica cayó a 0 DJs"
        assert p['analysis_count'] == 43, "se perdió el histórico de análisis"
