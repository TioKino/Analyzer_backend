"""El backfill de huella que SI funciona en el Mac App Store y en movil.

El backfill de siempre (`POST /backfill-fingerprint`) lo calcula el CLIENTE con
su `fpcalc` y manda 4 KB. Es lo barato, y por eso es lo que corre en Windows y
en el DMG de macOS. Pero necesita ese binario, y hay dos plataformas donde no
puede haberlo:

  - build del Mac App Store: el sandbox impide que el subproceso abra ficheros.
  - movil: no hay binario que empaquetar.

Son justo las dos con mas legado sin huella, o sea fuera de la memoria
colectiva para siempre. `POST /backfill-audio` cierra ese hueco por el otro
lado: el audio lo tienen ellas, asi que lo suben y la huella la saca el
servidor — sin reanalizar, sin AudD y sin tocar bpm/key.

Lo que estos tests atan, y por que:

  1. Las CUATRO respuestas existen y se distinguen. Un solo «no se pudo» haria
     indistinguible «este fichero no esta analizado» (subirlo seria un analisis
     completo con su AudD detras) de «fpcalc no puede con este audio» (no
     reintentar nunca) de «otro aparato se adelanto» (ya esta hecho).
  2. Una fila sin analizar NO se analiza. Es la linea entre un backfill y una
     factura sorpresa.
  3. La escritura es un UPDATE de dos columnas, no un `save_track`: reescribir
     la fila entera bombearia `analyzed_at`, que es EL numero que decide si el
     hueco de huella es legado o una via abierta.
  4. `/acoustic/pending` parte el lote en tres cubos, por lo mismo del punto 1.
  5. El cache-hit por FILENAME de /analyze tambien cura la huella. Sin eso, la
     cura no llegaba al caso mas comun: subir otra vez el mismo fichero con el
     mismo nombre devolvia antes de que el camino por huella pudiera mirarla.

    pytest test_backfill_audio.py -v
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient

import main
from main import app, db as main_db

client = TestClient(app)

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


def _audio(relleno=b'\xff\xfb\x90\x00'):
    """Bytes que pasan el minimo de tamano. No es audio real: fpcalc no se
    ejecuta en ningun test (se sustituye `_attach_acoustic` donde hace falta)."""
    return relleno * 1000


def _subir(datos, nombre='x.mp3'):
    return client.post(
        '/backfill-audio',
        files={'file': (nombre, io.BytesIO(datos), 'audio/mpeg')},
    )


def _sembrar(fp, *, chromaprint=None, analyzed_at='2024-01-15T10:00:00'):
    main_db.save_track({
        'id': fp, 'filename': f'{fp}.mp3', 'artist': 'A', 'title': 'T',
        'duration': 300.0, 'bpm': 128.0, 'key': 'A min', 'camelot': '8A',
        'energy_dj': 6, 'genre': 'Techno', 'track_type': 'peak',
        'fingerprint': fp, 'chromaprint': chromaprint,
        'analyzed_at': analyzed_at,
    })


def _md5_de(datos):
    import hashlib
    return hashlib.md5(datos).hexdigest()


# ============================================================================
# LAS CUATRO RESPUESTAS
# ============================================================================

def test_un_audio_sin_analizar_NO_se_analiza():
    """La linea entre un backfill y una factura sorpresa.

    Analizar aqui gastaria CPU y posiblemente AudD por una peticion que el
    usuario cree gratis — y en un movil recien instalado eso seria la
    biblioteca ENTERA.
    """
    datos = _audio(b'\x01\x02\x03\x04')
    r = _subir(datos)
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'no_analizado'
    assert body['ok'] is False
    assert main_db.get_track_by_fingerprint(body['fingerprint']) is None, (
        'el endpoint no puede dar de alta tracks'
    )


def test_una_fila_que_ya_tiene_huella_no_se_recalcula(monkeypatch):
    datos = _audio(b'\x05\x06\x07\x08')
    _sembrar(_md5_de(datos), chromaprint='YA_LA_TENIA')

    llamadas = []
    monkeypatch.setattr(main, '_attach_acoustic',
                        lambda d, p: llamadas.append(p))

    body = _subir(datos).json()
    assert body['status'] == 'ya_tenia'
    assert body['ok'] is True
    assert llamadas == [], 'fpcalc son ~2s de CPU: no se pagan para reescribir lo mismo'


def test_fpcalc_que_no_puede_con_el_fichero_lo_dice(monkeypatch):
    """`_attach_acoustic` es best-effort y se traga la excepcion. Devolver un OK
    vacio haria que el cliente resubiera los mismos megas en cada tanda."""
    datos = _audio(b'\x09\x0a\x0b\x0c')
    _sembrar(_md5_de(datos))
    monkeypatch.setattr(main, '_attach_acoustic', lambda d, p: None)

    body = _subir(datos).json()
    assert body['status'] == 'sin_huella'
    assert body['ok'] is False


def test_curar_escribe_chromaprint_y_cluster(monkeypatch):
    datos = _audio(b'\x0d\x0e\x0f\x10')
    fp = _md5_de(datos)
    _sembrar(fp)

    def _falso(track_data, audio_path):
        assert os.path.exists(audio_path), 'el audio tiene que estar en disco'
        track_data['chromaprint'] = 'HUELLA_NUEVA'
        track_data['acoustic_id'] = 'cluster_1'

    monkeypatch.setattr(main, '_attach_acoustic', _falso)

    body = _subir(datos).json()
    assert body['status'] == 'curada'
    assert body['ok'] is True
    assert body['acoustic_id'] == 'cluster_1'

    fila = main_db.get_track_by_fingerprint(fp)
    assert fila['chromaprint'] == 'HUELLA_NUEVA'
    assert fila['acoustic_id'] == 'cluster_1'


def test_curar_NO_bombea_analyzed_at(monkeypatch):
    """El motivo por el que se escribe con `backfill_track_fingerprint` y no con
    `save_track`: `newest_without` decide entre dos arreglos OPUESTOS (viejo =
    ampliar backfill, reciente = tapar una via abierta), y un backfill masivo
    que refresque las fechas convertiria todo el legado en «via abierta»."""
    datos = _audio(b'\x11\x12\x13\x14')
    fp = _md5_de(datos)
    _sembrar(fp, analyzed_at='2024-03-01T09:00:00')

    monkeypatch.setattr(main, '_attach_acoustic', lambda d, p: d.update(
        {'chromaprint': 'H', 'acoustic_id': 'c'}))

    assert _subir(datos).json()['status'] == 'curada'
    assert main_db.get_track_by_fingerprint(fp)['analyzed_at'] == '2024-03-01T09:00:00'


def test_no_toca_el_analisis(monkeypatch):
    """Ni bpm, ni key, ni genero, ni el analysis_json. Es un backfill."""
    datos = _audio(b'\x15\x16\x17\x18')
    fp = _md5_de(datos)
    _sembrar(fp)
    antes = main_db.get_track_by_fingerprint(fp)

    monkeypatch.setattr(main, '_attach_acoustic', lambda d, p: d.update(
        {'chromaprint': 'H', 'acoustic_id': 'c'}))
    _subir(datos)

    despues = main_db.get_track_by_fingerprint(fp)
    for campo in ('bpm', 'key', 'camelot', 'genre', 'energy_dj', 'track_type',
                  'filename', 'analysis_json'):
        assert despues[campo] == antes[campo], f'{campo} cambio en un backfill'


def test_el_temporal_se_borra_siempre(monkeypatch):
    """El disco de Render es pequeno y esto sube ficheros enteros."""
    vistos = []

    def _falso(track_data, audio_path):
        vistos.append(audio_path)

    datos = _audio(b'\x19\x1a\x1b\x1c')
    _sembrar(_md5_de(datos))
    monkeypatch.setattr(main, '_attach_acoustic', _falso)
    _subir(datos)

    assert vistos and not os.path.exists(vistos[0])


def test_un_fichero_diminuto_se_rechaza():
    r = _subir(b'123')
    assert r.status_code == 400


def test_appledouble_se_rechaza():
    """macOS crea `._nombre.mp3` en volumenes NTFS/FAT. No son audio."""
    r = _subir(_audio(), nombre='._x.mp3')
    assert r.status_code == 400


# ============================================================================
# /acoustic/pending — TRES cubos, no dos
# ============================================================================

def test_pending_separa_curable_de_no_analizado():
    _sembrar('pend_sin', chromaprint=None)
    _sembrar('pend_con', chromaprint='H')

    r = client.post('/acoustic/pending', json={
        'fingerprints': ['pend_sin', 'pend_con', 'pend_nunca_visto'],
    })
    assert r.status_code == 200
    body = r.json()
    assert body['curable'] == ['pend_sin']
    assert body['with_chromaprint'] == ['pend_con']
    assert body['not_analyzed'] == ['pend_nunca_visto'], (
        'sin este cubo, un movil recien instalado subiria su biblioteca entera '
        'creyendo que rellena huecos'
    )
    assert body['total'] == 3


def test_pending_tiene_tope_de_lote():
    r = client.post('/acoustic/pending',
                    json={'fingerprints': [f'f{i}' for i in range(501)]})
    assert r.status_code == 400


def test_pending_con_lote_vacio_no_revienta():
    r = client.post('/acoustic/pending', json={'fingerprints': []})
    assert r.status_code == 200
    assert r.json()['total'] == 0


def test_pending_encuentra_las_filas_legacy_por_id():
    """En los registros antiguos `tracks.id` ES el MD5 y `fingerprint` puede
    venir vacio. Mirar una sola columna dejaria fuera media biblioteca."""
    main_db.save_track({
        'id': 'legacy_id_1', 'filename': 'l.mp3', 'duration': 200.0,
        'bpm': 120.0, 'energy_dj': 5, 'genre': 'House', 'track_type': 'peak',
    })
    body = client.post('/acoustic/pending',
                       json={'fingerprints': ['legacy_id_1']}).json()
    assert body['curable'] == ['legacy_id_1']


# ============================================================================
# EL AGUJERO DEL CACHE-HIT POR FILENAME
# ============================================================================

def test_el_cache_hit_por_filename_tambien_cura_la_huella():
    """Este atajo exige nombre igual Y huella igual — o sea, el mismo fichero
    subido otra vez, que es lo que hace todo el mundo al reimportar su carpeta.
    Devolvia ANTES de que el camino por huella pudiera mirar el chromaprint, asi
    que la fila legada solo se curaba si el fichero habia cambiado de nombre."""
    src = _src('main.py')
    i = src.index('# Limpiar el tmp_path creado durante el upload streaming')
    tramo = src[max(0, i - 1200):i]
    assert "_fila = db._row_to_dict(existing) or {}" in tramo
    assert "if not _fila.get('chromaprint'):" in tramo
    assert '_attach_acoustic(_fila, tmp_path)' in tramo
    assert 'db.backfill_track_fingerprint(' in tramo


def test_el_backfill_por_audio_no_llama_a_audd():
    src = _src('main.py')
    i = src.index('async def backfill_audio_endpoint(')
    fn = src[i:src.index('# ==================== CACHE-LOOKUP / ARTWORK', i)]
    for prohibido in ('audd', 'analyze_audio', 'generate_preview_snippet',
                      'db.save_track('):
        assert prohibido not in fn, (
            f'{prohibido} no pinta nada en un backfill: encarece una peticion '
            f'que el usuario cree gratis'
        )
