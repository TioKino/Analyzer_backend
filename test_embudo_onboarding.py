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

from routes.admin_panel import (  # noqa: E402
    _FUNNEL_DEFAULT,
    _FUNNEL_MOBILE,
    _funnel_steps_for,
)


def _eventos(pasos):
    return [e for e, _ in pasos]


# ============================================================================
# EL DENOMINADOR
# ============================================================================

def test_onboarding_shown_va_ANTES_de_completed_en_los_dos_embudos():
    for pasos in (_FUNNEL_DEFAULT, _FUNNEL_MOBILE):
        ev = _eventos(pasos)
        assert 'onboarding_shown' in ev, 'falta el denominador real'
        assert ev.index('onboarding_shown') < ev.index('onboarding_completed')


def test_el_paso_anterior_a_completed_es_shown_y_no_app_opened():
    """Lo que hace que el porcentaje signifique algo: `drop_from_prev` se
    calcula contra el paso ANTERIOR, asi que con `app_opened` delante medias
    la finalizacion contra gente que nunca vio la pantalla."""
    for pasos in (_FUNNEL_DEFAULT, _FUNNEL_MOBILE):
        ev = _eventos(pasos)
        i = ev.index('onboarding_completed')
        assert ev[i - 1] == 'onboarding_shown'


def test_movil_y_desktop_comparten_el_arranque():
    # Si solo se arregla uno, los dos porcentajes dejan de ser comparables y
    # la conclusion «movil va peor que desktop» pasa a no significar nada.
    assert _eventos(_FUNNEL_MOBILE)[:3] == _eventos(_FUNNEL_DEFAULT)[:3]


def test_el_selector_de_plataforma_sigue_funcionando():
    assert _funnel_steps_for('mobile') is _FUNNEL_MOBILE
    assert _funnel_steps_for('ios') is _FUNNEL_MOBILE
    assert _funnel_steps_for('android') is _FUNNEL_MOBILE
    assert _funnel_steps_for('macos') is _FUNNEL_DEFAULT
    assert _funnel_steps_for(None) is _FUNNEL_DEFAULT


def test_movil_sigue_sin_import_como_paso_obligatorio():
    """Se corrigio el 2026-08-19: en movil la musica llega por sync, y medir
    import como paso pintaba como fuga a gente haciendo lo correcto. Añadir un
    paso no puede haberlo deshecho."""
    ev = _eventos(_FUNNEL_MOBILE)
    assert 'import_started' not in ev
    assert 'import_completed' not in ev


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
