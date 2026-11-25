"""El onboarding se mide contra los que LO VIERON, no contra los que abrieron.

Lectura de produccion del 2026-08-27:

    [mobile] Abrio la app: 144 -> Completo onboarding: 67 (46.5%)

y eso se leyo como «movil pierde a la mitad en el onboarding». Pero
`app_opened` se emite **una vez por dispositivo PARA SIEMPRE**, mientras que el
onboarding solo lo ve quien instala nuevo (`mobile_onboarding_shown_v1`, flag en
prefs). O sea que el denominador incluye a todo el que ya tenia la app
instalada y por tanto NO PODIA completarlo.

Es el mismo fallo que el «2 de 141» de `device_linked` — que esta dos pasos mas
abajo en la misma lista, con su aviso escrito al lado.

El cliente ya emitia `onboarding_shown` y `onboarding_skipped` desde hace
tiempo; el embudo simplemente no los usaba.

    pytest test_embudo_onboarding.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embudo  # noqa: E402

# Desde el 2026-11-25 los pasos viven en `embudo.py` (v3) y son los MISMOS
# para las dos plataformas: cada aparato los da con los eventos de la suya.
# Lo que estos tests guardaban se sigue guardando: la bienvenida se mide
# contra quien LA VIO, las dos plataformas son comparables y en el movil
# importar no es obligatorio.


# ============================================================================
# EL DENOMINADOR
# ============================================================================

def test_la_bienvenida_vista_va_ANTES_de_la_pasada():
    assert 'bienvenida_vista' in embudo.PASOS, 'falta el denominador real'
    assert (embudo.PASOS.index('bienvenida_vista')
            < embudo.PASOS.index('bienvenida_pasada'))


def test_el_paso_anterior_a_pasarla_es_verla_y_no_abrir():
    """`caen_pct` se calcula contra el paso ANTERIOR: con «abrio» delante se
    mediria la finalizacion contra gente que nunca vio la pantalla."""
    i = embudo.PASOS.index('bienvenida_pasada')
    assert embudo.PASOS[i - 1] == 'bienvenida_vista'


def test_movil_y_escritorio_comparten_los_pasos():
    # Si cada uno tuviera los suyos, «el movil va peor que el escritorio»
    # dejaria de significar nada. Los nombres son comunes; lo que cambia son
    # los eventos con que se da cada uno (`pasos_directos`).
    for g in embudo.GRUPOS:
        assert set(embudo.pasos_directos(g, set())) == set(embudo.PASOS)


def test_el_grupo_sale_de_la_plataforma():
    assert embudo.grupo_de('ios') == 'movil'
    assert embudo.grupo_de('android') == 'movil'
    for p in ('macos', 'macos-mas', 'macos-dmg', 'windows', 'linux'):
        assert embudo.grupo_de(p) == 'escritorio', p
    assert embudo.grupo_de(None) is None
    assert embudo.grupo_de('web') is None, 'lo que no se sabe no cae en un grupo'


def test_en_el_movil_importar_no_es_obligatorio():
    """Se corrigio el 2026-08-19: en movil la musica llega por sync o se usa
    Escuchar. Quien solo identifico un tema llega al valor sin importar."""
    d = embudo.pasos_directos('movil', {'abrio', 'escucho', 'identifico'})
    assert d['empezo'] and d['valor']
    lejos = embudo.paso_mas_lejano(d)
    assert embudo.PASOS[lejos] == 'valor'


def test_en_el_movil_saltar_la_bienvenida_tambien_es_pasarla():
    d = embudo.pasos_directos('movil', {'onboarding_shown', 'onboarding_skipped'})
    assert d['bienvenida_pasada']


# ============================================================================
# EL REPARTO DE LOS QUE NO TERMINAN
# ============================================================================

def _reparto(shown, done, skip):
    """La misma cuenta que hace el endpoint."""
    return {
        'shown': shown,
        'completed': done,
        'skipped': skip,
        'abandoned': max(shown - done - skip, 0),
        'completion_pct': round(100.0 * done / shown, 1) if shown else None,
    }


def test_saltar_y_abandonar_son_cosas_DISTINTAS():
    """Piden arreglos opuestos: quien salta tomo una decision; quien abandona
    se atasco. Mezclarlos en «no completaron» es el fallo de `failedAudd`."""
    r = _reparto(shown=100, done=46, skip=40)
    assert r['skipped'] == 40
    assert r['abandoned'] == 14
    assert r['completion_pct'] == 46.0


def test_el_porcentaje_es_sobre_los_que_LO_VIERON():
    # 67 de 144 que abrieron = 46,5% (el numero enganoso).
    # 67 de 80 que lo vieron  = 83,8% (el numero real).
    assert _reparto(shown=80, done=67, skip=10)['completion_pct'] == 83.8


def test_sin_nadie_que_lo_haya_visto_el_pct_es_None_y_no_cero():
    """0% se lee como «nadie lo completa». None se lee como «no hay datos», que
    es lo que de verdad pasa cuando aun no lo ha visto nadie."""
    assert _reparto(shown=0, done=0, skip=0)['completion_pct'] is None


def test_abandoned_no_puede_salir_NEGATIVO():
    """Los tres son conteos independientes de dispositivos unicos en una
    ventana movil de 30 dias, asi que descuadran en los bordes: un aparato que
    vio el onboarding hace 31 dias y lo completo hace 29 suma en `completed` y
    no en `shown`. Un negativo ahi se leeria como un bug de contabilidad."""
    assert _reparto(shown=10, done=9, skip=5)['abandoned'] == 0


def test_el_endpoint_lo_devuelve():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'routes', 'admin_panel.py')
    with open(ruta, encoding='utf-8') as f:
        src = f.read()
    assert '"onboarding": onboarding,' in src
    assert '"abandoned": max(_ob_shown - _ob_done - _ob_skip, 0),' in src
