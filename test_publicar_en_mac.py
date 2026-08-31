"""«Publicar esta biblioteca» devolvia 403 en TODOS los Mac.

El bug, entero:

`Analyzer#109` afino `PlatformUtils.platformHeader` de `macos` a
`macos-dmg` / `macos-mas`, para poder separar el Mac App Store del DMG — son
los dos sitios donde la huella acustica se comporta distinto (en MAS el
sandbox impide que `fpcalc` abra ficheros).

El cliente manda ese valor como `device_type` al publicar. Y el guard del
endpoint comparaba contra una lista escrita a mano con el vocabulario VIEJO:

    if req.device_type not in ("windows", "macos", "linux"):
        raise HTTPException(403, ...)

`macos-dmg` no esta en esa lista -> 403 -> el cliente ve un no-200, devuelve
`null`, y la UI dice «no se ha podido publicar, intentalo de nuevo». Ni un
error en los logs del cliente que dijera por que.

Se colo en la 2.9.10 —la release donde Publicar era la feature estrella— y
solo se vio probandolo en un Mac real. Windows y Linux nunca fallaron, porque
su valor no cambio: por eso ningun test lo cazo.

El otro consumidor roto por lo mismo, mas silencioso: `/admin/stats` contaba
`desktop_users` con la misma lista literal, asi que los Mac dejaron de contar
como escritorio — y tampoco caian en `mobile_users`. Desaparecian del reparto.

    pytest test_publicar_en_mac.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation import (  # noqa: E402
    is_desktop_platform, is_mobile_platform, platform_family,
    DESKTOP_PLATFORMS, MOBILE_PLATFORMS, ALLOWED_PLATFORMS,
)

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


# ============================================================================
# EL CASO QUE FALLABA
# ============================================================================

def test_el_DMG_de_mac_puede_publicar():
    """El valor que manda un Mac con motor local. Era el 403."""
    assert is_desktop_platform('macos-dmg')


def test_el_MAC_APP_STORE_puede_publicar():
    """La otra build de Mac. Mismo 403."""
    assert is_desktop_platform('macos-mas')


def test_las_plataformas_que_nunca_fallaron_siguen_pasando():
    """Windows y Linux no cambiaron de valor, y por eso nadie vio el bug.
    Quedan fijadas para que un arreglo del Mac no las rompa a ellas."""
    for p in ('windows', 'linux', 'macos'):
        assert is_desktop_platform(p), p


def test_el_movil_NO_puede_publicar():
    """La regla de producto sigue en pie: el movil es el aparato que se
    reinstala y se queda sin espacio, y no puede imponer su estado al resto."""
    for p in ('android', 'ios', 'mobile'):
        assert not is_desktop_platform(p), p


def test_TODA_plataforma_permitida_cae_en_una_familia():
    """El test que habria cazado el bug al hacer el cambio de #109.

    `ALLOWED_PLATFORMS` es la lista de valores que el cliente puede mandar.
    Si alguno no pertenece a ninguna familia, hay un consumidor en alguna
    parte que lo va a dejar fuera sin dar ningun error — que es exactamente lo
    que paso con `macos-dmg`.
    """
    huerfanas = [p for p in ALLOWED_PLATFORMS if platform_family(p) is None]
    assert huerfanas == [], f'sin familia: {huerfanas}'


def test_las_familias_no_se_solapan():
    assert DESKTOP_PLATFORMS & MOBILE_PLATFORMS == frozenset()


def test_un_valor_desconocido_no_cae_en_ningun_cubo():
    """None es «no lo se». Meterlo en escritorio por defecto convertiria un
    hueco en un dato — y ademas dejaria publicar a cualquiera."""
    for p in (None, '', 'web', 'playstation', 'unknown'):
        assert platform_family(p) is None, p
        assert not is_desktop_platform(p), p
        assert not is_mobile_platform(p), p


def test_tolera_mayusculas_y_espacios():
    assert is_desktop_platform('  MacOS-DMG  ')
    assert is_mobile_platform('Android')


# ============================================================================
# EL CABLEADO: UNA SOLA LISTA
# ============================================================================

def test_publish_usa_el_helper_y_no_una_lista_a_mano():
    src = _src('sync_endpoints.py')
    assert 'if not is_desktop_platform(req.device_type):' in src
    assert 'not in ("windows", "macos", "linux")' not in src, (
        'vuelve a estar la lista literal que causo el 403'
    )


def test_el_403_dice_QUE_valor_rechazo():
    """El mensaje anterior repetia la lista permitida y no decia que habia
    llegado. Con `macos-dmg` delante, el bug se habria visto en el primer
    vistazo al detalle del error."""
    src = _src('sync_endpoints.py')
    i = src.index('if not is_desktop_platform(req.device_type):')
    assert 'req.device_type!r' in src[i:i + 500]


def test_admin_stats_tambien_usa_el_helper():
    src = _src('routes/admin_panel.py')
    assert 'if is_desktop_platform(r["device_type"])' in src
    assert 'if is_mobile_platform(r["device_type"])' in src
    assert '("desktop", "macos", "windows", "linux")' not in src


def test_no_quedan_listas_de_plataforma_a_mano_en_el_backend():
    """La regla: si un sitio pregunta «esto es escritorio?», viene a
    `validation`. Este test es la red para el proximo valor nuevo."""
    sospechosas = []
    for nombre in ('sync_endpoints.py', 'routes/admin_panel.py', 'main.py'):
        src = _src(nombre)
        for patron in ('"windows", "macos"', "'windows', 'macos'",
                       '"macos", "windows"', "'macos', 'windows'"):
            if patron in src:
                sospechosas.append((nombre, patron))
    assert sospechosas == [], sospechosas
