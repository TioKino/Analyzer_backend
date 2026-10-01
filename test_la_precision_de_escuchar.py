"""La precision de Escuchar y como casa Shazam (2026-10-01).

Desde ese dia el acierto se guarda solo en el historial: ya no hay boton
GUARDAR, asi que `listen_saved` deja de llegar y la precision no puede salir
de «guardadas frente a equivocadas». Sale de los aciertos (`listen_result`
con `found`, que mandan todas las versiones) frente a los «No es este»
(`listen_wrong`).

Y `shazam_senal`: con *Pure NRG* (Astral Projection), que Shazam no tiene en
su catalogo, el iPhone del owner recibio cinco temas de trance parecidos y
AudD dijo «no lo conozco» las tres veces. Antes de filtrar nada hay que ver
en que se distinguen los equivocados de los buenos: el desvio de tono
(`shazam_skew`), el segundo en que caso (`shazam_offset_s`) y si casaron
varias grabaciones (`shazam_candidatos`).

    pytest test_la_precision_de_escuchar.py -v
"""

import json
import uuid

import main
from routes import admin_panel

# Una variante que no usa ningun otro test: la BD de eventos es compartida.
_PROPS = {'motor': 'shazam', 'clip_s': 6, 'envio': 'ffmpeg',
          'audd_tras_s': 19}
_VARIANTE = 'shazam+6s+ffmpeg+audd19s'


def _evento(dev, nombre, **props):
    main.db.log_event(device_id=dev, event_name=nombre,
                      props=json.dumps({**_PROPS, **props}), platform='ios')


def test_la_precision_sale_de_los_aciertos_y_los_no_es_este():
    dev = f'movil-{uuid.uuid4().hex[:8]}'
    for _ in range(3):
        _evento(dev, 'listen_result', outcome='found', resuelto_por='shazam')
    _evento(dev, 'listen_result', outcome='no_match')
    _evento(dev, 'listen_wrong', resuelto_por='shazam')
    m = admin_panel._escuchar_segun_el_movil()
    v = m['por_variante'][_VARIANTE]
    assert v['equivocadas'] == 1
    assert v['precision'] == round(1 - 1 / 3, 3)
    assert 0 <= m['precision'] <= 1


def test_sin_aciertos_la_precision_es_none():
    m = admin_panel._escuchar_segun_el_movil()
    sin = [v for v in m['por_variante'].values()
           if not v['por_desenlace'].get('found')]
    assert all(v['precision'] is None for v in sin)


def test_la_senal_de_shazam_separa_aciertos_y_equivocados():
    antes = admin_panel._escuchar_segun_el_movil()['shazam_senal']
    dev = f'movil-{uuid.uuid4().hex[:8]}'
    _evento(dev, 'listen_result', outcome='found', resuelto_por='shazam',
            shazam_skew=0.0004, shazam_offset_s=85.2, shazam_candidatos=1)
    _evento(dev, 'listen_result', outcome='found', resuelto_por='shazam',
            shazam_skew=-0.031, shazam_offset_s=12.0, shazam_candidatos=2)
    # De AudD, o sin senal (Android, version vieja): no cuentan.
    _evento(dev, 'listen_result', outcome='found', resuelto_por='audd',
            shazam_skew=0.5)
    _evento(dev, 'listen_result', outcome='found', resuelto_por='shazam')
    _evento(dev, 'listen_wrong', resuelto_por='shazam', shazam_skew=-0.031,
            shazam_offset_s=12.0, shazam_candidatos=2)
    d = admin_panel._escuchar_segun_el_movil()['shazam_senal']
    assert d['aciertos']['n'] - antes['aciertos']['n'] == 2
    assert d['equivocados']['n'] - antes['equivocados']['n'] == 1
    assert (d['aciertos']['varios_candidatos']
            - antes['aciertos']['varios_candidatos']) == 1
    assert d['equivocados']['skew_abs_p50'] >= 0, 'en valor absoluto'
    assert isinstance(d['aciertos']['skew_abs_p90'], float)
