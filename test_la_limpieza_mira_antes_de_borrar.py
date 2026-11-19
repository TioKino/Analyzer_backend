"""La limpieza del servidor MIRA antes de borrar (2026-11-19).

El owner quiso limpiar la base de datos y el disco de Render de lo que
dejaron años de pruebas. Primero un diagnóstico que no escribe nada
(`limpieza.py`, `GET /admin/limpieza`): este test monta una base con una fila
de cada clase y unos ficheros rotos y huérfanos, y comprueba que cada uno cae
donde toca, que lo bueno no sale en ninguna y que no se escribe nada.

    pytest test_la_limpieza_mira_antes_de_borrar.py -v
"""
import hashlib
import json
import os
import sqlite3

import limpieza

MP3 = b'ID3' + b'\x00' * 6000
JPG = b'\xFF\xD8\xFF\xE0' + b'\x00' * 2000


def _base(tmp_path):
    ruta = tmp_path / 'analysis.db'
    conn = sqlite3.connect(ruta)
    conn.execute(
        'CREATE TABLE tracks (id TEXT PRIMARY KEY, filename TEXT, artist TEXT,'
        ' title TEXT, duration REAL, bpm REAL, analyzed_at TEXT,'
        ' fingerprint TEXT, acoustic_id TEXT, analysis_json TEXT)')

    def fila(fp, artista='DJ', titulo=None, bpm=128.0, dur=360.0, js=None,
             cluster=None, fichero=None):
        conn.execute(
            'INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?)',
            (fp, fichero or f'{fp}.mp3', artista,
             titulo or f'Tema {fp.rstrip("0123456789")}', dur,
             bpm, '2026-05-01T10:00:00', fp, cluster,
             json.dumps(js or {'bpm': bpm})))

    fila('bueno1')
    fila('bueno2', cluster='c-ok')
    fila('bueno3', cluster='c-ok')
    fila('ilegible', bpm=0, js={'analysis_status': 'failed', 'fallo': 'ilegible'})
    fila('error', bpm=0, js={'analysis_status': 'failed', 'fallo': 'error'})
    fila('escuchar', bpm=0, js={'analysis_status': 'recognize_only'})
    fila('vacia', artista='', titulo=' ', bpm=0)
    fila('corto', dur=6.0)
    fila('largo', dur=3600.0)
    fila('prueba', titulo='Pista de prueba desfase', fichero='pista_de_prueba.mp3')
    for i, nombre in enumerate(['Uno', 'Dos', 'Tres', 'Cuatro']):
        fila(f'silencio{i}', titulo=nombre, cluster='c-silencio')
    # Una sesión subida pista a pista, y un álbum sin nombres (posible).
    for i in range(1, 5):
        fila(f'spinning{i}', artista='DJ Toni', titulo=f'Sesion Spinning Pista {i}')
        fila(f'album{i}', artista='Grupo', titulo=f'Track {i:02d}')
    # Con número pero solo dos: no es nada.
    fila('rmx1', artista='X', titulo='Tema 1')
    fila('rmx2', artista='X', titulo='Tema 2')
    conn.commit()
    return conn


def _carpetas(tmp_path):
    prev = tmp_path / 'previews'
    art = tmp_path / 'artwork'
    prev.mkdir()
    art.mkdir()
    (prev / 'bueno1.mp3').write_bytes(MP3)
    (prev / 'nadie.mp3').write_bytes(MP3)  # huérfano
    (prev / 'bueno2.mp3').write_bytes(b'')  # vacío
    (prev / 'bueno3.mp3').write_bytes(b'<html>' + b'x' * 6000)  # no es mp3
    (prev / 'a.mp3.tmp').write_bytes(MP3)  # a medias
    (art / 'bueno1.jpg').write_bytes(JPG)
    (art / 'otro.jpg').write_bytes(JPG)  # huérfano
    return str(prev), str(art)


def _huella_de(carpeta):
    h = hashlib.md5()
    for raiz, _, ficheros in sorted(os.walk(carpeta)):
        for f in sorted(ficheros):
            p = os.path.join(raiz, f)
            h.update(f.encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()


def test_cada_clase_cae_donde_toca_y_no_se_escribe_nada(tmp_path):
    conn = _base(tmp_path)
    prev, art = _carpetas(tmp_path)
    antes = _huella_de(tmp_path)

    d = limpieza.diagnostico(conn, prev, art,
                             database_path=str(tmp_path / 'analysis.db'))
    t = d['tracks']

    def huellas(clase):
        return {e['huella'] for e in t[clase]['ejemplos']}

    assert t['total_tracks'] == 24
    assert t['detecciones_de_escuchar'] == 1
    assert huellas('fallidos_ilegibles') == {'ilegible'}
    assert huellas('fallidos_otros') == {'error'}
    assert huellas('sin_datos') == {'vacia'}
    assert huellas('muy_cortos') == {'corto'}
    assert huellas('muy_largos') == {'largo'}
    assert huellas('nombres_de_prueba') == {'prueba'}
    hc = t['huella_compartida']
    assert hc['clusters'] == 1 and hc['n'] == 4
    assert hc['ejemplos'][0]['cluster'] == 'c-silencio'
    ps = t['partes_de_sesion']
    assert ps['sesiones'] == 1 and ps['n'] == 4
    assert ps['ejemplos'][0]['artista'] == 'DJ Toni'
    pp = t['partes_posibles']
    assert pp['sesiones'] == 1 and pp['n'] == 4, 'el álbum sin nombres va aparte'
    for clase in ('fallidos_ilegibles', 'fallidos_otros', 'sin_datos',
                  'muy_cortos', 'muy_largos', 'nombres_de_prueba'):
        assert not (huellas(clase) & {'bueno1', 'bueno2', 'bueno3', 'escuchar'})

    p = d['previews']
    assert {e['fichero'] for e in p['rotos']['ejemplos']} == {
        'bueno2.mp3', 'bueno3.mp3', 'a.mp3.tmp'}
    assert {e['fichero'] for e in p['huerfanos']['ejemplos']} == {'nadie.mp3'}
    assert p['huerfanos']['bytes'] == len(MP3)
    a = d['portadas']
    assert a['rotos']['n'] == 0
    assert {e['fichero'] for e in a['huerfanos']['ejemplos']} == {'otro.jpg'}
    assert 'libre' in d['disco']

    conn.close()
    assert _huella_de(tmp_path) == antes, 'el diagnóstico no escribe nada'


def test_los_nombres_de_prueba_no_pillan_temas_de_verdad():
    assert limpieza.es_nombre_de_prueba('Test Tone 1kHz')
    assert limpieza.es_nombre_de_prueba(None, 'pista_de_prueba_lame.mp3')
    assert not limpieza.es_nombre_de_prueba('Testarossa', 'Contest')
    assert not limpieza.es_nombre_de_prueba('Jamie Jones', 'Panic')
    for de_verdad in ('A Test of Patience', 'I Love You, You Big Dummy',
                      'Beep Me 911', 'Di Rigori Armato Il Seno'):
        assert not limpieza.es_nombre_de_prueba(de_verdad), de_verdad


def test_la_palabra_de_sesion_es_la_del_cliente():
    assert limpieza.dice_que_es_una_sesion('Sesión de spinning')
    assert limpieza.dice_que_es_una_sesion('Live @ Fabric')
    assert not limpieza.dice_que_es_una_sesion('Tema (Remix)')
    assert not limpieza.dice_que_es_una_sesion('Box Set')
    assert limpieza.patron_numerado('Pista 03') == 'pista #'
    assert limpieza.patron_numerado('Panic') is None


def test_la_ruta_pide_el_secreto_y_esta_montada():
    src = open('routes/admin_panel.py', encoding='utf-8').read()
    i = src.index('@admin_panel_router.get("/limpieza")')
    cuerpo = src[i:i + 900]
    assert 'await _verify_admin_secret(request)' in cuerpo
    assert 'solo_lectura=True' in cuerpo, 'el diagnóstico abre la base en solo lectura'
    assert '_pesado(' in cuerpo, 'recorre la tabla entera: de uno en uno'


def test_el_borrado_solo_se_lleva_lo_decidido(tmp_path):
    import time
    conn = _base(tmp_path)
    prev, art = _carpetas(tmp_path)
    viejo = os.path.join(prev, 'viejo.mp3.tmp')
    open(viejo, 'wb').write(b'x')
    os.utime(viejo, (time.time() - 7200, time.time() - 7200))
    plan = limpieza.plan_de_borrado(conn, prev, art)
    assert plan['temas_sin_datos'] == 1
    assert plan['temas_de_sesiones'] == 4
    assert plan['previews_vacias'] == 2, 'el vacío y el .tmp viejo'
    hecho = limpieza.borrar(conn, prev, art)
    assert hecho['filas_borradas'] == 5 and hecho['ficheros_borrados'] == 2
    quedan = {r[0] for r in conn.execute('SELECT id FROM tracks')}
    assert 'vacia' not in quedan and not any(q.startswith('spinning') for q in quedan)
    for sigue in ('ilegible', 'error', 'escuchar', 'album1', 'corto', 'prueba'):
        assert sigue in quedan, sigue
    assert os.path.exists(os.path.join(prev, 'a.mp3.tmp')), 'un .tmp reciente se queda'
    assert os.path.exists(os.path.join(prev, 'nadie.mp3')), 'los huérfanos se quedan'
