"""Las huellas que no dicen nada no juntan temas distintos (2026-11-19).

El diagnóstico de la limpieza dio 282 grupos acústicos con temas de nombres
distintos: samples de un segundo (con `_MIN_OVERLAP` = 8 casaban entre sí) y
canciones con la huella de un fichero mudo o truncado. Este test ata las dos
mitades del arreglo: que no vuelva a pasar (`huella_que_no_dice_nada`, en
`find_acoustic_cluster`) y que lo ya juntado se separe por nombre sin borrar
nada (`limpieza.separar`).

    pytest test_las_huellas_que_no_dicen_nada.py -v
"""
import random
import sqlite3

import limpieza
from acoustic_fingerprint import huella_que_no_dice_nada


def _huella(n, semilla):
    rnd = random.Random(semilla)
    return [rnd.getrandbits(32) for _ in range(n)]


def test_lo_que_no_dice_nada():
    assert huella_que_no_dice_nada(_huella(8, 1)), 'un segundo de audio'
    assert huella_que_no_dice_nada([7] * 1500), 'silencio: siempre lo mismo'
    assert huella_que_no_dice_nada(_huella(1500, 1), duration=12.0), 'un sample'
    assert not huella_que_no_dice_nada(_huella(1500, 1), duration=360.0)
    assert not huella_que_no_dice_nada(_huella(1500, 1))


def test_un_sample_no_entra_en_ningun_grupo(tmp_path, monkeypatch):
    import database
    db = database.AnalysisDB(str(tmp_path / 'a.db'))
    corta = _huella(10, 2)
    from acoustic_fingerprint import encode_raw
    conn = sqlite3.connect(tmp_path / 'a.db')
    conn.execute('INSERT INTO tracks (id, fingerprint, chromaprint, acoustic_id,'
                 ' duration) VALUES (?,?,?,?,?)',
                 ('k1', 'k1', encode_raw(corta), 'grupo-k', 1.0))
    conn.commit()
    conn.close()
    assert db.find_acoustic_cluster(corta, 1.0) is None
    assert db.resolve_acoustic_cluster(corta, 1.0) != 'grupo-k'


def _base(tmp_path):
    conn = sqlite3.connect(tmp_path / 'b.db')
    conn.execute('CREATE TABLE tracks (id TEXT PRIMARY KEY, fingerprint TEXT,'
                 ' artist TEXT, title TEXT, filename TEXT, acoustic_id TEXT)')
    filas = [
        ('a1', 'Uno', 'Success'), ('a2', 'Uno', 'Success'),
        ('a3', 'Dos', 'Girl Of My Dreams'), ('a4', 'Tres', 'Sharpen The Saw'),
        ('a5', 'Cuatro', 'Serato Recording 1'),
    ]
    for tid, art, tit in filas:
        conn.execute('INSERT INTO tracks VALUES (?,?,?,?,?,?)',
                     (tid, 'fp-' + tid, art, tit, tid + '.mp3', 'falso'))
    # Un grupo de verdad: el mismo tema en dos ficheros.
    for tid in ('b1', 'b2'):
        conn.execute('INSERT INTO tracks VALUES (?,?,?,?,?,?)',
                     (tid, 'fp-' + tid, 'X', 'Panic', tid + '.mp3', 'bueno'))
    conn.commit()
    return conn


def test_se_separa_por_nombre_y_el_mayor_se_queda_el_grupo(tmp_path):
    conn = _base(tmp_path)
    plan = limpieza.plan_de_separacion(conn)
    assert plan['grupos'] == 1 and plan['temas_que_cambian'] == 3
    hecho = limpieza.separar(conn)
    assert hecho['hecho']
    ids = dict(conn.execute('SELECT id, acoustic_id FROM tracks').fetchall())
    assert ids['a1'] == ids['a2'] == 'falso', 'las copias siguen juntas'
    assert len({ids['a3'], ids['a4'], ids['a5'], 'falso'}) == 4
    assert ids['b1'] == ids['b2'] == 'bueno', 'un grupo de verdad no se toca'
    assert conn.execute('SELECT COUNT(*) FROM tracks').fetchone()[0] == 7
    assert limpieza.plan_de_separacion(conn)['grupos'] == 0, 'una vez basta'
