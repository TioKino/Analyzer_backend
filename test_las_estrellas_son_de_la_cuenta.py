"""Tus estrellas son de tu CUENTA, no del aparato donde las pusiste.

Hasta el 2026-09-26 la valoración se guardaba y se leía por aparato: lo que
valorabas en el ordenador no salía en tu móvil vinculado, y si valorabas el
mismo tema en los dos contabas DOS veces en la media de la comunidad.

Hoy se lee por cuenta (`aparatos_de`, que main.py cablea a sync.db), valorar
desde otro aparato tuyo sustituye la anterior, quitarla la quita de todos, y la
media cuenta una por cuenta.

    pytest test_las_estrellas_son_de_la_cuenta.py -v
"""

import pytest

from database import AnalysisDB

FP = 'c' * 32
CUENTAS = {'mac': 'u1', 'movil': 'u1', 'otro': 'u2'}


@pytest.fixture
def db(tmp_path):
    d = AnalysisDB(db_path=str(tmp_path / 'analysis.db'))
    d.cuentas_de = lambda ids: {i: CUENTAS[i] for i in ids if i in CUENTAS}
    d.aparatos_de = lambda dev: sorted(
        a for a, u in CUENTAS.items() if u == CUENTAS.get(dev)) or [dev]
    return d


def test_lo_que_valoras_en_el_ordenador_sale_en_el_movil(db):
    db.rate_track(FP, 'mac', 4)
    assert db.get_my_rating(FP, 'movil') == 4
    assert db.get_my_ratings_batch([FP], 'movil') == {FP: 4}
    assert db.get_my_rating(FP, 'otro') == 0, 'otra cuenta no lo ve'


def test_valorar_desde_otro_aparato_tuyo_sustituye(db):
    db.rate_track(FP, 'mac', 2)
    r = db.rate_track(FP, 'movil', 5)
    assert r == {'avg_rating': 5.0, 'total_ratings': 1}
    assert db.get_my_rating(FP, 'mac') == 5


def test_quitarla_la_quita_de_todos_tus_aparatos(db):
    db.rate_track(FP, 'mac', 4)
    db.rate_track(FP, 'movil', 0)
    assert db.get_my_rating(FP, 'mac') == 0


def test_la_media_cuenta_una_por_cuenta(db):
    db.rate_track(FP, 'otro', 1)
    db.rate_track(FP, 'mac', 5)
    assert db.get_track_popularity(FP)['total_ratings'] == 2
    assert db.get_track_popularity(FP)['avg_rating'] == 3.0


def test_las_duplicadas_de_antes_cuentan_una(db):
    # Valoradas desde dos aparatos de la misma cuenta ANTES de este cambio.
    for dev, nota, fecha in (('mac', 1, '2026-01-01'), ('movil', 5, '2026-02-01')):
        db.conn.execute('INSERT INTO track_ratings (fingerprint, device_id, '
                        'rating, rated_at) VALUES (?, ?, ?, ?)',
                        (FP, dev, nota, fecha))
    db.conn.commit()
    r = db.rate_track(FP, 'otro', 3)
    assert r == {'avg_rating': 4.0, 'total_ratings': 2}, 'u1 = 5 (la más nueva), u2 = 3'
    assert db.get_my_rating(FP, 'mac') == 5


def test_sin_cuentas_cada_aparato_es_el_suyo(tmp_path):
    d = AnalysisDB(db_path=str(tmp_path / 'b.db'))
    d.rate_track(FP, 'mac', 4)
    assert d.get_my_rating(FP, 'movil') == 0
