"""El `OR` entre `fingerprint` e `id` NO usa indice en SQLite.

Todo lo batch de community busca por huella, y `tracks.fingerprint` era la
UNICA columna sin indice: cada peticion recorria las 122.103 filas, y el
cliente manda ~23 lotes seguidos al cargar la libreria. De ahi los
`TimeoutException` de 12 s en cadena de `/community/my-ratings/batch` y el 200
con cuerpo vacio (medido 2026-09-23).

Y el indice no basta: con el `OR` cruzando dos columnas, SQLite hace recorrido
completo igual. Por eso se parte en DOS queries.
"""
import re
import sqlite3
from pathlib import Path

FUENTE = Path(__file__).resolve().parent.parent / 'database.py'


def test_no_queda_ningun_or_entre_huella_e_id():
    texto = FUENTE.read_text(encoding='utf-8')
    # Se busca el patron, no un sitio concreto: estaba en CUATRO y la gracia es
    # que no vuelva por ninguno.
    assert 'OR id IN' not in texto, (
        'volvio el OR entre fingerprint e id: eso es recorrido completo'
    )


def test_el_indice_de_fingerprint_existe():
    texto = FUENTE.read_text(encoding='utf-8')
    assert 'idx_tracks_fingerprint ON tracks(fingerprint)' in texto


def test_el_ayudante_deduplica_por_id_y_mira_las_dos_columnas():
    """Sobre una BD de juguete: mismo resultado que el OR, incluido el legado.

    En los registros antiguos el **id ES el MD5**, asi que buscar por una sola
    columna deja fuera media biblioteca historica. Y una fila puede aparecer
    por las dos: sin deduplicar se procesaria dos veces.
    """
    con = sqlite3.connect(':memory:')
    con.row_factory = sqlite3.Row
    c = con.cursor()
    c.execute('CREATE TABLE tracks (id TEXT PRIMARY KEY, fingerprint TEXT, '
              'acoustic_id TEXT)')
    c.executemany('INSERT INTO tracks VALUES (?,?,?)', [
        ('id_moderno', 'fp_moderno', 'ac1'),   # fila normal
        ('md5_legacy', None, 'ac2'),           # legado: el id ES el md5
        ('id_sinac', 'fp_sinac', None),        # sin cluster
        ('id_ambas', 'md5_legacy', 'ac3'),     # su fingerprint es el id de otra
    ])

    def por_or(fps, extra=''):
        m = ','.join('?' * len(fps))
        cond = f' AND {extra}' if extra else ''
        c.execute(f'SELECT id FROM tracks WHERE (fingerprint IN ({m}) '
                  f'OR id IN ({m})){cond}', fps + fps)
        return {r['id'] for r in c.fetchall()}

    def por_dos(fps, extra=''):
        m = ','.join('?' * len(fps))
        cond = f' AND {extra}' if extra else ''
        vistas = []
        for col in ('fingerprint', 'id'):
            c.execute(f'SELECT id FROM tracks WHERE {col} IN ({m}){cond}', fps)
            for r in c.fetchall():
                if r['id'] not in vistas:
                    vistas.append(r['id'])
        return set(vistas)

    casos = [
        ['fp_moderno'],
        ['md5_legacy'],                        # legado: entra por `id`
        ['fp_sinac'],
        ['md5_legacy', 'fp_moderno'],
        ['md5_legacy', 'fp_sinac', 'fp_moderno'],
        ['no_existe'],
    ]
    for fps in casos:
        for extra in ('', 'acoustic_id IS NOT NULL'):
            assert por_or(fps, extra) == por_dos(fps, extra), (fps, extra)


def test_las_columnas_pedidas_incluyen_id():
    """El ayudante deduplica por `id`; sin el en el SELECT daria KeyError."""
    texto = FUENTE.read_text(encoding='utf-8')
    for m in re.finditer(r"_tracks_por_huella_o_id\(\s*c,\s*'([^']+)'", texto):
        assert 'id' in [x.strip() for x in m.group(1).split(',')], m.group(1)
