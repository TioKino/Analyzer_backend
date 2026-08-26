"""Los cues comunitarios cruzan entre DJs — o no cruzan nada.

Durante anios el cliente subia y leia estos cues con su `track.id`, que es
MD5("nombre|tamanio"). Aqui `tracks.id = tracks.fingerprint` (MD5 del
CONTENIDO), asi que ese `track.id` no existe en ninguna tabla:
`fingerprints_in_cluster()` no encontraba fila, no resolvia el cluster acustico
y devolvia la clave pelada. Cada usuario escribia en su propio compartimento.

Nadie lo noto porque el circuito era coherente CONSIGO MISMO: tus cues te
volvian a ti. Para que cruzaran harian falta dos DJs con ficheros de identico
nombre Y tamanio, y en produccion `collision_groups` es 0 entre 317
dispositivos. O sea: no cruzo ni una vez, y ningun test lo decia porque todos
probaban el circuito cerrado.

Estos tests prueban lo que el producto PROMETE, no lo que el codigo hace: dos
DJs, dos ficheros distintos del mismo tema, los cues de uno visibles para el
otro. Y que la transicion desde la clave vieja no pierde nada.

    pytest test_community_cues_key.py -v
"""

import pytest

from fastapi.testclient import TestClient

from main import app, db

client = TestClient(app)


# ============================================================================
# HELPERS
# ============================================================================
#
# OJO CON LA BD: este fichero comparte `analysis.db` con toda la suite.
#
# El patron `os.environ['DATABASE_PATH'] = tempfile...` dentro de una fixture
# —que esta copiado en varios test_*.py de aqui— NO aisla nada: `conftest.py`
# ya fijo la ruta antes de la coleccion, y `main` (con su `db`) se importa la
# primera vez que cualquier modulo lo pide. Cuando la fixture corre, la
# conexion lleva rato abierta contra otra ruta.
#
# Se probo, y el resultado fue que estos tests tumbaban
# `test_dedup_render_fallback.py`: los dos usaban `'a'*32` como fingerprint y
# las filas sembradas aqui aparecian alli. Verde por separado, rojo la suite.
#
# Asi que: claves con prefijo propio (imposible chocar) y limpieza de lo
# sembrado al terminar el modulo.

_P = 'cuekeytest_'

# Dos ficheros distintos del MISMO tema: distinto codec, distinto tamanio, asi
# que distinto MD5 de contenido. Lo que los une es el cluster acustico.
FP_MP3 = _P + 'fp_mp3'
FP_FLAC = _P + 'fp_flac'
CLUSTER = _P + 'cluster'

# La clave vieja: el `track.id` del cliente. Nunca estuvo en `tracks`.
LEGACY_TRACK_ID = _P + 'legacy_trackid'

# Todos los device_id llevan el prefijo por el mismo motivo.
DEV_A = _P + 'device_A'
DEV_B = _P + 'device_B'
DEV_VICTIMA = _P + 'device_VICTIMA'


@pytest.fixture(autouse=True, scope='module')
def _limpiar_al_salir():
    """No dejar rastro en la BD compartida."""
    yield
    c = db.conn.cursor()
    c.execute('DELETE FROM tracks WHERE id LIKE ?', (_P + '%',))
    c.execute('DELETE FROM community_cues WHERE fingerprint LIKE ? '
              'OR device_id LIKE ?', (_P + '%', _P + '%'))
    db.conn.commit()


def _sembrar_cluster():
    """Mete los dos ficheros en `tracks` con el MISMO `acoustic_id`.

    Es lo que hace `_attach_acoustic` en produccion al analizar. Sin esto no
    hay cluster y cada fingerprint va por su cuenta.
    """
    for fp in (FP_MP3, FP_FLAC):
        db.save_track({
            'id': fp,
            'fingerprint': fp,
            'filename': fp + '.mp3',
            'duration': 300.0,
            'bpm': 128.0,
            'key': 'Am',
            'camelot': '8A',
            'energy_dj': 7,
            'genre': 'techno',
            'track_type': 'original',
            'acoustic_id': CLUSTER,
        })


def _subir(key, device, posiciones, legacy_key=None, cue_type='drop'):
    body = {
        'fingerprint': key,
        'device_id': device,
        'cues': [{'type': cue_type, 'position_ms': p} for p in posiciones],
    }
    if legacy_key is not None:
        body['legacy_key'] = legacy_key
    r = client.post('/community/cues', json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _leer(key, also=None):
    url = f'/community/cues/{key}'
    if also:
        url += f'?also={also}'
    r = client.get(url)
    assert r.status_code == 200, r.text
    return r.json()


def _filas(key, device=None):
    """Filas crudas en la tabla para una clave (y opcionalmente un device)."""
    c = db.conn.cursor()
    if device:
        c.execute(
            'SELECT COUNT(*) n FROM community_cues '
            'WHERE fingerprint = ? AND device_id = ?', (key, device))
    else:
        c.execute(
            'SELECT COUNT(*) n FROM community_cues WHERE fingerprint = ?',
            (key,))
    return c.fetchone()['n']


@pytest.fixture(autouse=True)
def _limpiar():
    # Solo lo NUESTRO: un `DELETE FROM community_cues` a secas se llevaria por
    # delante lo que hayan sembrado otros modulos de la suite.
    _sembrar_cluster()
    c = db.conn.cursor()
    c.execute('DELETE FROM community_cues WHERE fingerprint LIKE ? '
              'OR device_id LIKE ?', (_P + '%', _P + '%'))
    db.conn.commit()
    yield


# ============================================================================
# LO QUE EL PRODUCTO PROMETE
# ============================================================================

class TestCruceEntreDJs:

    def test_dos_djs_con_ficheros_distintos_comparten_zona(self):
        """El caso de uso entero, en un test.

        DJ A tiene el mp3, DJ B el flac. Marcan el drop casi en el mismo sitio.
        Al abrir su fichero, cada uno tiene que ver la zona de los DOS.
        """
        _subir(FP_MP3, DEV_A, [90000])
        _subir(FP_FLAC, DEV_B, [90400])

        data = _leer(FP_MP3)

        assert data['total_contributors'] == 2, \
            'los dos ficheros son el mismo audio: tienen que sumar'
        assert len(data['zones']) == 1
        assert data['zones'][0]['dj_count'] == 2

    def test_el_que_llega_despues_tambien_lo_ve(self):
        # Simetria: no puede depender de quien subio primero.
        _subir(FP_MP3, DEV_A, [90000])
        _subir(FP_FLAC, DEV_B, [90400])

        assert _leer(FP_FLAC)['zones'][0]['dj_count'] == 2

    def test_con_la_clave_VIEJA_no_cruza_nada(self):
        """El bug, escrito como test para que no vuelva.

        Si el cliente sube con su `track.id`, el backend no puede resolverlo a
        ningun cluster: la fila no existe en `tracks`. Queda aislado.
        """
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        _subir(FP_FLAC, DEV_B, [90400])

        # El DJ B no ve nada del A: hacen falta 2 DJs para una zona.
        assert _leer(FP_FLAC)['zones'] == []
        # Y el A tampoco ve al B.
        assert _leer(LEGACY_TRACK_ID)['zones'] == []


# ============================================================================
# LA TRANSICION NO PIERDE NADA
# ============================================================================

class TestClaveLegada:

    def test_also_recupera_lo_subido_con_la_clave_vieja(self):
        # Dos devices del MISMO usuario que subieron bajo la clave vieja: eso
        # es lo unico que llegaba a formar zona antes del cambio.
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        _subir(LEGACY_TRACK_ID, DEV_B, [90400])

        # Sin `also`, el cliente nuevo no las ve...
        assert _leer(FP_MP3)['zones'] == []
        # ...con `also`, si. Nada desaparece mientras no vuelva a tocar el track.
        assert _leer(FP_MP3, also=LEGACY_TRACK_ID)['zones'][0]['dj_count'] == 2

    def test_also_suma_a_lo_del_cluster_no_lo_sustituye(self):
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        _subir(FP_FLAC, DEV_B, [90400])

        data = _leer(FP_MP3, also=LEGACY_TRACK_ID)
        assert data['zones'][0]['dj_count'] == 2, \
            'tiene que ver lo viejo suyo Y lo nuevo del cluster'

    def test_also_con_una_clave_inventada_no_rompe(self):
        _subir(FP_MP3, DEV_A, [90000])
        _subir(FP_FLAC, DEV_B, [90400])

        data = _leer(FP_MP3, also='no_existe_esta_clave')
        assert data['zones'][0]['dj_count'] == 2


class TestMigracionAlEscribir:

    def test_subir_con_legacy_key_borra_las_filas_viejas_de_ese_device(self):
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        assert _filas(LEGACY_TRACK_ID, DEV_A) == 1

        _subir(FP_MP3, DEV_A, [90000], legacy_key=LEGACY_TRACK_ID)

        assert _filas(LEGACY_TRACK_ID, DEV_A) == 0, \
            'las filas viejas ya estan representadas por las nuevas'
        assert _filas(FP_MP3, DEV_A) == 1

    def test_mover_un_cue_no_deja_la_posicion_vieja_colgada(self):
        """El motivo REAL de borrar, y no solo la limpieza.

        Sin el borrado, mover un cue deja la posicion antigua bajo la clave
        vieja: el DJ pide `?also=` y ve SU zona por duplicado, en dos sitios de
        la onda, sin haber marcado nada dos veces.
        """
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        _subir(LEGACY_TRACK_ID, DEV_B, [90100])
        # device_A mueve su drop 30 s mas adelante y ya va con huella.
        _subir(FP_MP3, DEV_A, [120000], legacy_key=LEGACY_TRACK_ID)
        _subir(FP_FLAC, DEV_B, [120200], legacy_key=LEGACY_TRACK_ID)

        zonas = _leer(FP_MP3, also=LEGACY_TRACK_ID)['zones']

        assert len(zonas) == 1, f'zona duplicada: {zonas}'
        assert zonas[0]['start'] >= 118.0, 'quedo la posicion vieja'

    def test_el_borrado_NO_toca_a_otros_devices(self):
        # La clave vieja es del cliente y no esta verificada. El borrado tiene
        # que estar acotado al device que llama, o cualquiera podria vaciar los
        # cues de otro mandando su clave.
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        _subir(LEGACY_TRACK_ID, DEV_VICTIMA, [90000])

        _subir(FP_MP3, DEV_A, [90000], legacy_key=LEGACY_TRACK_ID)

        assert _filas(LEGACY_TRACK_ID, DEV_VICTIMA) == 1, \
            'un device ha borrado datos de otro'

    def test_legacy_key_igual_a_la_clave_no_borra_lo_recien_escrito(self):
        """Un track SIN huella manda las dos claves iguales.

        Si el borrado no lo detectara, subiria los cues y acto seguido los
        borraria: el DJ marca cues y no se guarda ninguno.
        """
        _subir(LEGACY_TRACK_ID, DEV_A, [90000],
               legacy_key=LEGACY_TRACK_ID)

        assert _filas(LEGACY_TRACK_ID, DEV_A) == 1

    def test_sin_legacy_key_no_se_borra_nada(self):
        # Cliente viejo: se comporta exactamente como antes.
        _subir(LEGACY_TRACK_ID, DEV_A, [90000])
        _subir(FP_MP3, DEV_A, [90000])

        assert _filas(LEGACY_TRACK_ID, DEV_A) == 1
        assert _filas(FP_MP3, DEV_A) == 1
