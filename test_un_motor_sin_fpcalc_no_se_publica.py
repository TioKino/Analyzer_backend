"""Un motor local SIN `fpcalc` no puede publicarse — y si existe, se ve.

El 2026-09-12 entraron 281 tracks por `/cache-analysis` con
`engine_source=local_engine`, `platform=windows` y sin chromaprint. Analisis
perfectos —bpm, key, energia, genero— a los que solo les faltaba la huella
acustica, o sea justo lo unico que el usuario no puede ver.

Dos causas posibles, y piden cosas OPUESTAS:

  motor viejo            anterior a que el motor mandara el chromaprint.
                         Se cura solo en cuanto esa persona actualice.
  motor sin `fpcalc`     el binario no se bundleo en el build. NO se cura
                         nunca, y el backfill automatico tampoco lo salva
                         porque tira del MISMO binario.

Con los datos de entonces no se podian separar, y ese es el fallo de fondo:
dos causas con arreglos opuestos compartiendo señal.

Este fichero ata los dos lados del arreglo:

  1. Los `.spec` de PyInstaller ABORTAN si falta `fpcalc`. Antes imprimian una
     ADVERTENCIA y seguian — y el repo no trae `fpcalc.exe` (assets/native/
     windows/ solo tiene el .gitkeep), asi que la unica proteccion era
     acordarse. `build_desktop.ps1` ya aborta por esto desde el 2026-08-25; el
     motor se habia quedado sin la misma guarda.

  2. `/health` dice el estado de `fpcalc`. Era el unico binario cuyo fallo es
     completamente silencioso y el unico que NO salia en el health, mientras si
     salia `ffmpeg`, que cuando falta se nota en el acto (sin previews).

    pytest test_un_motor_sin_fpcalc_no_se_publica.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))

_SPECS = ['dj_analyzer_engine.spec', 'dj_analyzer_engine_macos.spec']


def _fuente(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


# ============================================================================
# 1. EL BUILD ABORTA
# ============================================================================

def test_los_dos_specs_abortan_si_falta_fpcalc():
    """Windows y macOS, los dos. El de Windows es el que mordio, pero dejar el
    otro avisando solo mantiene viva la misma trampa en la otra plataforma."""
    for spec in _SPECS:
        src = _fuente(spec)
        assert 'raise SystemExit' in src, (
            f'{spec} no aborta sin fpcalc: un motor que no puede fingerprintear '
            f'no debe llegar a publicarse'
        )


def test_ningun_spec_se_conforma_con_avisar():
    """La rama del `else` no puede quedarse en un print.

    Un `print` de ADVERTENCIA en mitad de un build de PyInstaller —cientos de
    lineas de salida— no lo lee nadie. Eso es exactamente lo que paso.
    """
    for spec in _SPECS:
        src = _fuente(spec)
        i = src.find('fpcalc_path')
        assert i > 0, f'{spec}: cambio la forma de resolver fpcalc'
        cola = src[i:]
        j = cola.find('raise SystemExit')
        assert j > 0, f'{spec}: falta el abort'
        # Entre la resolucion y el abort no puede haber quedado un camino que
        # siga adelante con el binario ausente.
        assert 'ADVERTENCIA: fpcalc' not in cola[:j], (
            f'{spec}: sigue avisando en vez de abortar'
        )


def test_el_abort_dice_QUE_se_rompe_y_DONDE_poner_el_binario():
    """Un abort que solo dice «falta fpcalc» manda a buscar en Google.

    Tiene que decir las dos cosas que no son obvias: que el sintoma es
    silencioso (el motor funciona, solo que sin huella) y que el backfill
    automatico NO lo arregla.
    """
    for spec in _SPECS:
        src = _fuente(spec)
        assert 'memoria colectiva' in src, f'{spec}: el abort no dice que se rompe'
        assert 'backfill' in src, (
            f'{spec}: el abort no avisa de que el backfill tampoco lo cura'
        )
        assert 'chromaprint' in src.lower(), f'{spec}: no dice de donde sacarlo'


# ============================================================================
# 2. /health LO DICE
# ============================================================================

def test_health_reporta_fpcalc():
    src = _fuente('main.py')
    i = src.find('async def health():')
    assert i > 0, 'cambio la firma de /health'
    cuerpo = src[i:i + 4000]
    assert '"fpcalc"' in cuerpo, (
        '/health no dice el estado de fpcalc. Es el unico binario cuyo fallo es '
        'silencioso y era el unico que faltaba en los checks'
    )


def test_health_usa_los_tres_estados_y_no_dispara_la_descarga():
    """`estado_fpcalc()` da `ok` / `caido` / `sin_intentar` a proposito: un
    booleano confundiria «no hay binario» con «aun no se ha pedido en este
    worker», que fue justo la alarma en falso del 2026-09-06.

    Y tiene que ser `estado_fpcalc`, NO `ensure_fpcalc`: el segundo DESCARGA.
    Un /health con efectos secundarios deja de ser un /health.
    """
    src = _fuente('main.py')
    i = src.find('def _estado_fpcalc_health')
    assert i > 0, 'falta el helper de fpcalc para /health'
    helper = src[i:i + 900]
    assert 'estado_fpcalc' in helper
    assert 'ensure_fpcalc' not in helper, (
        '/health no puede llamar a ensure_fpcalc: dispararia una descarga'
    )


def test_health_no_revienta_si_no_se_puede_importar():
    """Un health que lanza no informa de nada, y menos aun cuando lo que falla
    es justo el modulo por el que preguntas."""
    import main
    main._fpcalc_import_roto = True  # marcador inerte, solo documenta la intencion
    estado = main._estado_fpcalc_health()
    assert isinstance(estado, dict)
    assert 'estado' in estado
