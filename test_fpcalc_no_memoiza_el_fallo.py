"""`ensure_fpcalc` memoizaba el FALLO, y eso apaga la memoria colectiva entera.

Medido el 2026-09-06: **682 tracks recientes con `engine_source='render'` y sin
chromaprint**, entrando a rafagas, mientras el log solo tenia tres
`fpcalc exit 2` sueltos. Los numeros no cuadran — y no cuadran porque no es que
fpcalc fallara 682 veces, es que **no se le llamo** 682 veces.

El codigo era:

    if _fpcalc_resolved:
        return _fpcalc_path      # None si el primer intento fallo
    ...
    _fpcalc_resolved = True      # se pone a True pasara lo que pasara

O sea que si el PRIMER intento del proceso no encontraba el binario —descarga
que expira, red de Render con hipo al arrancar, /data aun sin montar— todas las
llamadas siguientes devolvian None sin reintentar, durante toda la vida del
worker. Y el Procfile levanta UN worker: la huella apagada para todo el mundo
hasta el proximo deploy.

Y era invisible por partida doble:

  - el `logger.info` de exito solo se emitia si habia path, asi que el fallo no
    escribia NADA en `ensure_fpcalc`;
  - la unica linea (`fpcalc no disponible`, en `compute_raw_chromaprint`) ni
    lleva la palabra «error» ni es severidad error, o sea que no salia ni
    filtrando los logs de Render por «error».

    pytest test_fpcalc_no_memoiza_el_fallo.py -v
"""

import logging

import pytest

import acoustic_fingerprint as af


@pytest.fixture(autouse=True)
def limpio():
    """Estado global a cero antes y despues: es memoizacion por proceso."""
    af._fpcalc_path = None
    af._fpcalc_fallos = 0
    af._fpcalc_ultimo_intento = 0.0
    yield
    af._fpcalc_path = None
    af._fpcalc_fallos = 0
    af._fpcalc_ultimo_intento = 0.0


class _Reloj:
    """monotonic() controlado, para no dormir en los tests."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def avanzar(self, s):
        self.t += s


@pytest.fixture
def reloj(monkeypatch):
    r = _Reloj()
    monkeypatch.setattr(af.time, 'monotonic', r)
    return r


def _resolucion(monkeypatch, *resultados):
    """Encadena lo que devuelve la resolucion en llamadas sucesivas."""
    caja = {'i': 0}

    def _r():
        i = min(caja['i'], len(resultados) - 1)
        caja['i'] += 1
        return resultados[i]
    monkeypatch.setattr(af, '_resolve_existing_fpcalc', _r)
    monkeypatch.setattr(af, '_download_fpcalc', lambda: None)
    return caja


# ============================================================================
# LA REGRESION
# ============================================================================

def test_un_fallo_al_arrancar_NO_apaga_la_huella_para_siempre(
        monkeypatch, reloj):
    """EL bug. Primer intento falla, y al reintentar mas tarde tiene que
    recuperarse solo — sin esperar a un deploy."""
    caja = _resolucion(monkeypatch, None, '/data/bin/fpcalc')

    assert af.ensure_fpcalc() is None          # arranque con la red caida
    reloj.avanzar(af._FPCALC_REINTENTO_S + 1)
    assert af.ensure_fpcalc() == '/data/bin/fpcalc'
    assert caja['i'] == 2, 'no reintento'


def test_entre_reintentos_NO_machaca_la_descarga(monkeypatch, reloj):
    """El otro lado del mismo problema: reintentar en CADA `/analyze` podria
    disparar una descarga por track. Dentro de la ventana no se toca nada."""
    caja = _resolucion(monkeypatch, None)

    for _ in range(50):
        assert af.ensure_fpcalc() is None
    assert caja['i'] == 1, 'reintento dentro de la ventana de espera'

    reloj.avanzar(af._FPCALC_REINTENTO_S + 1)
    af.ensure_fpcalc()
    assert caja['i'] == 2


def test_el_exito_SI_se_cachea(monkeypatch, reloj):
    """Lo que la memoizacion venia a hacer y sigue haciendo: una vez resuelto,
    ni un `stat` mas por track."""
    caja = _resolucion(monkeypatch, '/usr/bin/fpcalc')

    for _ in range(20):
        assert af.ensure_fpcalc() == '/usr/bin/fpcalc'
    assert caja['i'] == 1


# ============================================================================
# QUE SE ENTERE ALGUIEN
# ============================================================================

def test_el_fallo_se_loguea_como_ERROR_y_dice_lo_que_significa(
        monkeypatch, reloj, caplog):
    """Antes el fallo no escribia NADA en `ensure_fpcalc` (el log de exito
    estaba dentro del `if path`). Ahora es `error` —no `warning`— porque es la
    memoria colectiva apagada, no un track que se pierde."""
    _resolucion(monkeypatch, None)
    with caplog.at_level(logging.DEBUG):
        af.ensure_fpcalc()
    assert any(r.levelno >= logging.ERROR for r in caplog.records), \
        'un apagon de la huella no puede ser un warning'
    assert 'NINGUN track' in caplog.text


def test_avisa_al_RECUPERARSE(monkeypatch, reloj, caplog):
    """Sin esta linea no hay forma de acotar la ventana del apagon, y por tanto
    de saber cuantos tracks hay que curar."""
    _resolucion(monkeypatch, None, '/data/bin/fpcalc')
    af.ensure_fpcalc()
    reloj.avanzar(af._FPCALC_REINTENTO_S + 1)
    with caplog.at_level(logging.DEBUG):
        af.ensure_fpcalc()
    assert 'RECUPERADO' in caplog.text


# ============================================================================
# Y QUE NO HAGA FALTA MIRAR LOS LOGS
# ============================================================================

def test_el_estado_sale_por_el_panel(monkeypatch, reloj):
    """La razon de que esto durase dias: el unico sitio donde constaba era el
    log, y el mensaje no lleva la palabra «error»."""
    _resolucion(monkeypatch, None)
    af.ensure_fpcalc()
    e = af.estado_fpcalc()
    assert e['estado'] == 'caido'
    assert e['disponible'] is False
    assert e['fallos_seguidos'] == 1
    assert 'memoria colectiva' in e['nota']


def test_NO_intentado_todavia_NO_es_caido(monkeypatch):
    """El falso positivo de la primera lectura real (2026-09-06): la alarma
    salio en rojo con `fallos_seguidos: 0`, o sea gritando «no hay binario»
    cuando lo que pasaba es que nadie habia pedido fpcalc en ese worker.

    Tras CADA deploy hay una ventana asi hasta el primer `/analyze`, o sea que
    el aviso saltaba en falso todo el rato. Y una alarma que grita en falso es
    peor que no tenerla: entrena a ignorarla, y la caida de verdad pasa
    desapercibida.

    «No» y «no lo se» no pueden compartir señal — la misma razon por la que en
    Duplicados se separan `clusters` de `without_cluster`.
    """
    monkeypatch.setattr(af, '_resolve_existing_fpcalc', lambda: '/data/bin/fpcalc')
    e = af.estado_fpcalc()
    assert e['estado'] == 'sin_intentar'
    assert e['fallos_seguidos'] == 0
    # Y no se queda en un encogimiento de hombros: dice si el binario esta.
    assert e['binario_en_disco'] is True


def test_sin_intentar_y_SIN_binario_en_disco(monkeypatch):
    """El matiz que hace util el tercer estado: aun no se ha pedido, pero
    cuando se pida no va a estar. No es una caida todavia, y casi."""
    monkeypatch.setattr(af, '_resolve_existing_fpcalc', lambda: None)
    e = af.estado_fpcalc()
    assert e['estado'] == 'sin_intentar'
    assert e['binario_en_disco'] is False


def test_resuelto_es_ok(monkeypatch, reloj):
    _resolucion(monkeypatch, '/usr/bin/fpcalc')
    af.ensure_fpcalc()
    e = af.estado_fpcalc()
    assert e['estado'] == 'ok'
    assert e['disponible'] is True


def test_el_estado_NO_dispara_una_descarga(monkeypatch, reloj):
    """Un endpoint admin que se lee para mirar no puede tener efectos.

    `estado_fpcalc` SI llama a `_resolve_existing_fpcalc` (solo `stat`/`which`),
    pero NUNCA a `_download_fpcalc`."""
    descargas = {'n': 0}

    def _no():
        descargas['n'] += 1
        return None
    monkeypatch.setattr(af, '_resolve_existing_fpcalc', lambda: None)
    monkeypatch.setattr(af, '_download_fpcalc', _no)
    for _ in range(5):
        af.estado_fpcalc()
    assert descargas['n'] == 0


def test_telemetry_lo_expone():
    import os
    aqui = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(aqui, 'routes/admin_panel.py'), encoding='utf-8') as f:
        src = f.read()
    assert '"fpcalc": _estado_fpcalc(),' in src
