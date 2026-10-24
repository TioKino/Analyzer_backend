"""El Resumen y el catálogo del panel admin (2026-10-24).

Owner: «el panel admin es un caos» y «un catálogo global de todos los tracks
analizados». Lo que se ata aquí:

  - El Resumen se calcula una vez al día y se LEE de una tabla; una medida que
    falla deja SUS números en `None` («falló») y no tumba el resto.
  - El catálogo sale de `tracks` (la memoria colectiva), sin nombre de fichero
    ni aparato, paginado, y cada consulta entra por un índice: la tabla tiene
    más de 120.000 filas con su `analysis_json` dentro.
  - `/admin/all-tracks`, que leía las bibliotecas personales de `sync.db` con
    nombres de fichero y aparatos, ya no existe.

    pytest test_el_resumen_y_el_catalogo.py -v
"""

import json
import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

import catalogo
import resumen_diario


# ── El Resumen ────────────────────────────────────────────────────────────


def test_una_medida_rota_deja_sus_numeros_en_none_y_sigue():
    def rota():
        raise RuntimeError('sync.db no responde')

    datos = resumen_diario.calcular({
        'uso': lambda: {'activos_30d': 312, 'volvio_alguna_vez_pct': 41.2},
        'biblioteca': rota,
        'audd': lambda: {'audd_30d_render': 1147, 'audd_30d_motor_local': 316},
        'basura': lambda: {'no_es_una_metrica': 1},
    })
    assert datos['activos_30d'] == 312
    assert datos['mediana_temas'] is None, 'falló: None, no 0'
    assert datos['audd_30d_total'] == 1463, 'el gasto real es la SUMA'
    assert 'no_es_una_metrica' not in datos
    assert any('biblioteca' in f for f in datos['_fallos'])
    assert set(resumen_diario.METRICAS) <= set(datos)


def test_la_adopcion_cuenta_tambien_lo_mas_nuevo():
    por_version = {'2.10.3': 2, '2.10.2': 6, '2.10.1': 1, '2.9.11': 1}
    # 2.10.3 son builds de prueba del owner: están al día, no atrasadas.
    assert resumen_diario.adopcion_pct(por_version, '2.10.2') == 80.0
    # 2.9.9 < 2.9.11 aunque como texto salga al revés.
    assert resumen_diario.adopcion_pct({'2.9.9': 1, '2.9.11': 1}, '2.9.11') == 50.0
    assert resumen_diario.adopcion_pct({}, '2.10.2') is None


def test_las_fotos_se_guardan_una_por_dia_y_salen_de_la_mas_nueva():
    conn = sqlite3.connect(':memory:')
    resumen_diario.guardar(conn, '2026-10-22', {'activos_30d': 300})
    resumen_diario.guardar(conn, '2026-10-24', {'activos_30d': 320})
    resumen_diario.guardar(conn, '2026-10-23', {'activos_30d': 310})
    resumen_diario.guardar(conn, '2026-10-24', {'activos_30d': 321})  # rehecha
    fotos = resumen_diario.ultimas(conn, 14)
    assert [f['dia'] for f in fotos] == ['2026-10-24', '2026-10-23', '2026-10-22']
    assert fotos[0]['datos']['activos_30d'] == 321


def test_el_panel_calcula_todas_las_medidas_sin_reventar():
    import main  # noqa: F401  (monta `main.db`)
    import routes.admin_panel as panel
    import sync_endpoints
    sync_endpoints._get_conn()  # las tablas de sync.db, como en Render
    datos = resumen_diario.calcular(panel._medidas_del_resumen())
    # En una BD de test vacía puede haber números a 0, pero ningún fallo: un
    # fallo aquí es un NameError o una columna que no existe.
    assert datos['_fallos'] == [], datos['_fallos']
    assert datos['version_actual'] == main.API_VERSION


def test_el_endpoint_del_resumen_lee_y_recalcula():
    import main
    import sync_endpoints
    sync_endpoints._get_conn()
    cliente = TestClient(main.app)
    r = cliente.post('/admin/resumen/recalcular')
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    assert cuerpo['hoy']['dia'] == resumen_diario.hoy()
    assert cuerpo['calculando'] is False
    assert cuerpo['metricas'] == list(resumen_diario.METRICAS)
    r2 = cliente.get('/admin/resumen')
    assert r2.status_code == 200
    assert r2.json()['hoy']['dia'] == resumen_diario.hoy()
    assert r2.json()['calculando'] is False, 'la de hoy ya está: no se rehace'


# ── El catálogo ───────────────────────────────────────────────────────────


@pytest.fixture()
def bd():
    """Una `analysis.db` con el esquema y los índices de verdad."""
    from database import AnalysisDB
    ruta = os.path.join(tempfile.mkdtemp(prefix='catalogo-'), 'analysis.db')
    AnalysisDB(ruta)
    conn = sqlite3.connect(ruta)
    filas = [
        # id/fp, artista, título, bpm, camelot, género, fecha, cluster, huella
        ('f1', 'Oxia', 'Domino', 125.0, '8A', 'Techno', '2026-10-01', 'c1', 'x'),
        ('f2', 'Oxia', 'Domino (Radio Edit)', 125.0, '8A', 'Techno', '2026-10-03', 'c1', 'x'),
        ('f3', 'Jam & Spoon', 'Age of Love', 132.5, '4A', 'Trance', '2026-10-02', None, None),
        ('f4', 'Nadie', 'Escuchado', 0.0, None, None, '2026-10-05', None, None),
        ('f5', 'Julian Jeweil', 'Air Conditionné', 126.0, '9A', 'Minimal', '2026-10-04', None, 'x'),
        ('f6', '100%_raro', 'Tema_con_guion', 128.0, '8A', 'Techno', '2026-09-30', None, None),
    ]
    for fp, a, t, bpm, cam, gen, fecha, cluster, huella in filas:
        conn.execute(
            "INSERT INTO tracks (id, filename, artist, title, bpm, camelot, genre, "
            "analyzed_at, fingerprint, acoustic_id, chromaprint, platform, "
            "analysis_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fp, f'/Users/alguien/{t}.mp3', a, t, bpm, cam, gen, fecha, fp,
             cluster, huella, 'windows',
             json.dumps({'bpm_source': 'rekordbox' if fp == 'f1' else 'analysis'})))
    conn.execute("INSERT INTO track_popularity (fingerprint, dj_count, last_analyzed) "
                 "VALUES ('c1', 7, '2026-10-03')")
    conn.execute("INSERT INTO track_popularity (fingerprint, dj_count, last_analyzed) "
                 "VALUES ('f3', 3, '2026-10-02')")
    conn.execute("INSERT INTO track_popularity (fingerprint, dj_count, last_analyzed) "
                 "VALUES ('f5', 1, '2026-10-04')")
    conn.commit()
    yield conn
    conn.close()


def test_solo_lo_analizado_y_lo_mas_nuevo_primero(bd):
    p = catalogo.pagina_del_catalogo(bd)
    assert [t['fingerprint'] for t in p['temas']] == ['f5', 'f2', 'f3', 'f1', 'f6']
    assert p['total'] == 5, 'la fila a BPM 0 (lo que siembra Escuchar) no cuenta'
    assert p['hay_mas'] is False


def test_sin_nombre_de_fichero_ni_aparato(bd):
    tema = catalogo.pagina_del_catalogo(bd)['temas'][0]
    texto = json.dumps(tema)
    assert 'filename' not in tema and 'file_name' not in tema
    assert '/Users/alguien' not in texto
    assert not any('device' in k for k in tema)


def test_cada_tema_dice_de_donde_salio_su_bpm_sus_djs_y_sus_versiones(bd):
    temas = {t['fingerprint']: t for t in catalogo.pagina_del_catalogo(
        bd, base_url='https://render', con_preview=lambda fp: fp == 'f1')['temas']}
    assert temas['f1']['bpm_source'] == 'rekordbox'
    assert temas['f1']['djs'] == 7, 'la popularidad va por el cluster'
    assert temas['f1']['versiones'] == 2
    assert temas['f3']['djs'] == 3 and temas['f3']['versiones'] == 1
    assert temas['f1']['con_huella'] is True and temas['f3']['con_huella'] is False
    assert temas['f1']['preview_url'] == 'https://render/preview/f1'
    assert temas['f3']['preview_url'] == ''
    assert temas['f1']['artwork_url'] == 'https://render/artwork/f1?online=0'


def test_paginado(bd):
    p0 = catalogo.pagina_del_catalogo(bd, por_pagina=2)
    p1 = catalogo.pagina_del_catalogo(bd, por_pagina=2, pagina=1)
    p2 = catalogo.pagina_del_catalogo(bd, por_pagina=2, pagina=2)
    assert [len(p0['temas']), len(p1['temas']), len(p2['temas'])] == [2, 2, 1]
    assert p0['hay_mas'] and p1['hay_mas'] and not p2['hay_mas']


def test_buscar_por_artista_o_titulo_y_un_comodin_es_texto(bd):
    p = catalogo.pagina_del_catalogo(bd, q='oxia')
    assert {t['fingerprint'] for t in p['temas']} == {'f1', 'f2'}
    assert p['total'] == 2
    assert {t['fingerprint'] for t in catalogo.pagina_del_catalogo(bd, q='age of')['temas']} == {'f3'}
    # «%» y «_» son letras, no comodines de LIKE.
    assert [t['fingerprint'] for t in catalogo.pagina_del_catalogo(bd, q='100%_')['temas']] == ['f6']
    # Como comodín, «o%o» casaría con Domino, Oxia… Como texto, con nada.
    assert catalogo.pagina_del_catalogo(bd, q='o%o')['temas'] == []
    assert catalogo.pagina_del_catalogo(bd, q='a_e')['temas'] == []


def test_filtros(bd):
    def fps(**kw):
        return [t['fingerprint'] for t in catalogo.pagina_del_catalogo(bd, **kw)['temas']]
    assert fps(genero='techno') == ['f2', 'f1', 'f6']
    assert fps(camelot='8a') == ['f2', 'f1', 'f6']
    assert fps(bpm_min=126, bpm_max=130) == ['f5', 'f6']
    assert catalogo.pagina_del_catalogo(bd, genero='techno')['total'] is None, \
        'con filtros el total no sale de un índice: no se da'


def test_las_versiones_del_mismo_audio_juntas(bd):
    p = catalogo.pagina_del_catalogo(bd, cluster='c1')
    assert [t['fingerprint'] for t in p['temas']] == ['f2', 'f1']
    assert p['total'] == 2


def test_por_numero_de_djs_una_fila_por_tema(bd):
    p = catalogo.pagina_del_catalogo(bd, orden='djs')
    fps = [t['fingerprint'] for t in p['temas']]
    # El cluster c1 (7 DJs) sale UNA vez, con su versión más nueva.
    assert fps == ['f2', 'f3', 'f5']
    assert p['total'] == 3
    # Buscando, también por DJs.
    assert [t['fingerprint'] for t in catalogo.pagina_del_catalogo(
        bd, orden='djs', q='a')['temas']][:2] in (['f2', 'f1'], ['f1', 'f2'])


def test_cada_consulta_entra_por_un_indice(bd):
    """Un «SCAN» de `tracks` sin «USING» es leer la tabla entera, con su
    `analysis_json`: segundos en caliente y 35 s en frío en Render."""
    planes = []

    class Espia:
        row_factory = None

        def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith('SELECT') and 'FROM tracks' in sql:
                plan = bd.execute('EXPLAIN QUERY PLAN ' + sql, params).fetchall()
                planes.append((sql, [str(r[-1]) for r in plan]))
            bd.row_factory = self.row_factory
            return bd.execute(sql, params)

    for kw in ({}, {'q': 'oxia'}, {'cluster': 'c1'}, {'orden': 'djs'},
               {'orden': 'djs', 'q': 'oxia'}, {'genero': 'techno'}):
        catalogo.pagina_del_catalogo(Espia(), **kw)
    assert len(planes) >= 8
    for sql, pasos in planes:
        for paso in pasos:
            if paso.startswith('SCAN') and 'subquery' not in paso:
                assert 'USING' in paso, (sql, pasos)


def test_el_endpoint_del_catalogo(bd):
    import main
    cliente = TestClient(main.app)
    r = cliente.get('/admin/catalogo', params={'q': 'x', 'orden': 'djs', 'por_pagina': 10})
    assert r.status_code == 200, r.text
    assert {'temas', 'pagina', 'por_pagina', 'total', 'hay_mas', 'orden'} <= set(r.json())
    assert cliente.get('/admin/catalogo', params={'orden': 'nombre'}).status_code == 422
    assert cliente.get('/admin/catalogo', params={'por_pagina': 1000}).status_code == 422


def test_all_tracks_ya_no_existe():
    import main
    rutas = {getattr(r, 'path', '') for r in main.app.routes}
    assert '/admin/all-tracks' not in rutas
    assert '/admin/catalogo' in rutas and '/admin/resumen' in rutas
