"""
`/admin/stats` tardaba 30 s y eso echaba al owner de su propio panel.

MEDIDO CONTRA PRODUCCION el 2026-09-08:

    /health          0,18 s
    /admin/stats    30,19 s      <- 165 veces mas lento
    /admin/errors    0,71 s

El servidor estaba sano. Solo ese endpoint tardaba. Y el sintoma con el que
aparecio no parecia tener nada que ver: **el dialogo del token de admin decia
«el backend no respondio a tiempo»**. Valida contra `/admin/stats` con dos
intentos de 20 s, asi que 30 s de escaneo rechazaban un token perfectamente
correcto. El gate fallaba por una razon ajena a lo que pregunta.

La causa: `_preview_exists` hacia un `os.path.isfile` POR TRACK, y con ~76.000
tracks unicos eso son 76.000 accesos al disco persistente de Render dentro de
una sola peticion. Ahora el directorio se lista una vez y se cachea unos
segundos.
"""
import os

import pytest

import routes.admin_panel as ap


@pytest.fixture(autouse=True)
def _cache_limpia():
    """El cache es global; sin esto un test se contamina con el anterior."""
    ap._previews_cache = set()
    ap._previews_cache_ts = 0.0
    yield
    ap._previews_cache = set()
    ap._previews_cache_ts = 0.0


class TestNoUnStatPorTrack:

    def test_mil_consultas_leen_el_directorio_UNA_vez(self, monkeypatch):
        """Es el arreglo entero. Si esto se rompe, /admin/stats vuelve a los
        30 s y el token vuelve a ser rechazado sin motivo."""
        listados = []

        def falso_listdir(path):
            listados.append(path)
            return [f'{i:032x}.mp3' for i in range(500)]

        monkeypatch.setattr(ap.os, 'listdir', falso_listdir)
        monkeypatch.setattr(
            ap.os.path, 'isfile',
            lambda p: pytest.fail('cayo al stat por fichero teniendo listado'),
        )

        for i in range(1000):
            ap._preview_exists(f'{i:032x}')

        assert len(listados) == 1, (
            f'leyo el directorio {len(listados)} veces; con 76.000 tracks eso '
            f'es lo que costaba 30 segundos'
        )

    def test_dice_la_verdad_sobre_quien_tiene_preview(self, monkeypatch):
        monkeypatch.setattr(
            ap.os, 'listdir', lambda p: ['aaaa.mp3', 'bbbb.mp3', 'ruido.txt'])
        assert ap._preview_exists('aaaa') is True
        assert ap._preview_exists('bbbb') is True
        assert ap._preview_exists('cccc') is False
        # Un fichero que no es .mp3 no cuenta como preview.
        assert ap._preview_exists('ruido') is False

    def test_sin_fingerprint_no_toca_el_disco(self, monkeypatch):
        monkeypatch.setattr(
            ap.os, 'listdir',
            lambda p: pytest.fail('listo el directorio sin fingerprint'))
        assert ap._preview_exists('') is False
        assert ap._preview_exists(None) is False


class TestSiElDirectorioFalla:
    """Lo unico que puede hacer daño de verdad."""

    def test_cae_al_stat_por_fichero_en_vez_de_MENTIR(self, monkeypatch):
        """Si el listado falla, devolver un set vacio diria «ningun track tiene
        preview»: una respuesta creible y falsa, que es peor que ser lento. El
        panel enseñaria `total_previews: 0` y alguien concluiria que se han
        borrado los previews.
        """
        def listdir_roto(path):
            raise OSError('disco no montado')

        consultados = []

        def falso_isfile(path):
            consultados.append(path)
            return path.endswith('siexiste.mp3')

        monkeypatch.setattr(ap.os, 'listdir', listdir_roto)
        monkeypatch.setattr(ap.os.path, 'isfile', falso_isfile)

        assert ap._preview_exists('siexiste') is True
        assert ap._preview_exists('noexiste') is False
        assert consultados, 'no cayo al stat por fichero'

    def test_un_fallo_no_deja_el_cache_envenenado(self, monkeypatch):
        """Si el disco vuelve, se vuelve a leer. Cachear el fallo es el mismo
        error que costo 682 tracks en `ensure_fpcalc`: alli un fallo al
        arrancar apagaba la huella hasta el siguiente deploy."""
        estado = {'roto': True}

        def listdir_intermitente(path):
            if estado['roto']:
                raise OSError('disco no montado')
            return ['bueno.mp3']

        monkeypatch.setattr(ap.os, 'listdir', listdir_intermitente)
        monkeypatch.setattr(ap.os.path, 'isfile', lambda p: False)

        assert ap._preview_exists('bueno') is False  # disco caido
        estado['roto'] = False
        assert ap._preview_exists('bueno') is True, (
            'el fallo se quedo cacheado y no reintento al volver el disco'
        )


def test_las_sesiones_se_iteran_sin_fetchall():
    """`fetchall()` sobre los payloads de sesiones se los trae todos a memoria
    de golpe, y ese patron ya tumbo produccion con un OOM (esta escrito en
    CLAUDE.md). Se lee del fuente porque la alternativa es levantar la BD
    entera para comprobar una linea."""
    ruta = os.path.join(os.path.dirname(__file__), 'routes', 'admin_panel.py')
    with open(ruta, encoding='utf-8') as f:
        src = f.read()
    i = src.index("data_type = 'session'")
    # La llamada y su `.fetchall()` caben de sobra en 200 caracteres.
    trozo = src[i:i + 200]
    assert '.fetchall()' not in trozo, (
        'las sesiones vuelven a leerse con fetchall(): se cargan todos los '
        'payloads en memoria a la vez'
    )
