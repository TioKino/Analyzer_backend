"""Cuánto acierta el DSP, medido contra lo que dicen los programas de DJ de la
MISMA huella (auditoría del flujo de análisis, tanda 3, 2026-10-06).

Hasta ese día lo importado de Rekordbox, Traktor y VirtualDJ se usaba para
SUSTITUIR el análisis, nunca para medirlo, y no había forma de saber si el
BPM y la tonalidad del DSP acertaban. Es la única verdad de referencia a mano:
el valor que el DJ ya tenía en su programa, sobre el mismo audio.

    pytest test_el_dsp_se_mide_contra_lo_importado.py -v
"""

import uuid
from datetime import datetime, timedelta

import pytest

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _tema(db, bpm=128.0, camelot='8A', bpm_source='analysis',
          key_source='analysis', hace_dias=1, legado=False):
    fp = uuid.uuid4().hex
    fila = {
        'id': fp, 'fingerprint': None if legado else fp,
        'filename': f'{fp}.mp3', 'bpm': bpm, 'bpm_source': bpm_source,
        'key': 'Am', 'camelot': camelot, 'key_source': key_source,
        'duration': 300, 'energy_dj': 6, 'genre': 'Techno',
        'track_type': 'peak_time',
        'analyzed_at': (datetime.utcnow() - timedelta(days=hace_dias)).isoformat(),
    }
    db.save_track(fila)
    return fp


def _importado(db, fp, programa='rekordbox', bpm=None, key=None, camelot=None,
               aparato='devA'):
    item = {'fingerprint': fp, 'source': programa}
    if bpm is not None:
        item['bpm'] = bpm
    if key is not None:
        item['key'], item['camelot'] = key, camelot
    db.guardar_lo_importado(aparato, [item])


def test_EL_CASO_el_bpm_y_la_tonalidad_medidos_frente_al_programa(db):
    bien = _tema(db, bpm=128.0, camelot='8A')
    _importado(db, bien, bpm=127.99, key='Am', camelot='8A')
    doble = _tema(db, bpm=170.0, camelot='8B')
    _importado(db, doble, bpm=85.0, key='Am', camelot='8A')
    otro = _tema(db, bpm=122.0, camelot='3A')
    _importado(db, otro, programa='traktor', bpm=126.0, key='Fm', camelot='4A')

    m = db.dsp_frente_a_lo_importado()
    assert m['bpm']['total'] == {'comparados': 3, 'igual': 1,
                                 'doble_o_mitad': 1, 'otro': 1}
    assert m['tonalidad']['total'] == {'comparados': 3, 'igual': 1,
                                       'relativa': 1, 'quinta': 1}
    assert m['bpm']['por_programa']['traktor'] == {'comparados': 1, 'igual': 0,
                                                   'otro': 1}


def test_lo_que_no_midio_el_dsp_no_se_cuenta(db):
    """Si la fila ya dice `id3` o `rekordbox`, el DSP no está ahí."""
    fp = _tema(db, bpm=128.0, bpm_source='id3', key_source='rekordbox')
    _importado(db, fp, bpm=100.0, key='Am', camelot='8A')
    m = db.dsp_frente_a_lo_importado()
    assert m['bpm']['total']['comparados'] == 0
    assert m['tonalidad']['total']['comparados'] == 0


def test_el_motor_local_tambien_es_dsp(db):
    fp = _tema(db, bpm=128.0, bpm_source='local_engine')
    _importado(db, fp, bpm=128.4)
    assert db.dsp_frente_a_lo_importado()['bpm']['total'] == {
        'comparados': 1, 'igual': 1}


def test_lo_reciente_aparte(db):
    """El DSP ha cambiado sin subir ANALYSIS_VERSION: lo de hoy se lee aparte."""
    viejo = _tema(db, bpm=140.0, hace_dias=200)
    _importado(db, viejo, bpm=128.0)
    nuevo = _tema(db, bpm=128.0, hace_dias=3)
    _importado(db, nuevo, bpm=128.0)
    m = db.dsp_frente_a_lo_importado(30)
    assert m['bpm']['total']['comparados'] == 2
    assert m['bpm']['recientes'] == {'comparados': 1, 'igual': 1}
    assert m['dias_recientes'] == 30


def test_un_valor_por_huella_aunque_voten_varios(db):
    fp = _tema(db, bpm=128.0)
    _importado(db, fp, bpm=128.0, aparato='devA')
    _importado(db, fp, bpm=128.0, aparato='devB')
    assert db.dsp_frente_a_lo_importado()['bpm']['total']['comparados'] == 1


def test_los_registros_antiguos_casan_por_id(db):
    fp = _tema(db, bpm=128.0, legado=True)
    _importado(db, fp, bpm=128.0)
    assert db.dsp_frente_a_lo_importado()['bpm']['total']['comparados'] == 1


def test_las_clases():
    c = AnalysisDB._clase_de_bpm
    assert c(128.0, 128.4) == 'igual'
    assert c(128.0, 129.5) == 'cerca'
    assert c(64.0, 128.0) == 'doble_o_mitad'
    assert c(256.5, 128.0) == 'doble_o_mitad'
    assert c(100.0, 128.0) == 'otro'
    assert c(0, 128.0) is None
    t = AnalysisDB._clase_de_tonalidad
    assert t('8A', '8a') == 'igual'
    assert t('8A', '8B') == 'relativa'
    assert t('12A', '1A') == 'quinta'
    assert t('1B', '12B') == 'quinta'
    assert t('8A', '3B') == 'otra'
    assert t(None, '8A') is None


def test_sale_en_el_panel_y_en_embudo():
    import os
    import main  # noqa: F401  (monta el singleton `main.db` que usa el panel)
    import routes.admin_panel as panel
    m = panel._dsp_frente_a_lo_importado()
    assert m is not None, 'None = embudo.sh dice «el panel falló»'
    assert {'bpm', 'tonalidad', 'dias_recientes'} <= set(m)
    with open(panel.__file__, encoding='utf-8') as fh:
        assert ('"dsp_frente_a_lo_importado": _dsp_frente_a_lo_importado()'
                in fh.read())
    embudo = os.path.join(os.path.dirname(__file__), '..', 'Analyzer',
                          'scripts', 'embudo.sh')
    if os.path.exists(embudo):
        with open(embudo, encoding='utf-8') as fh:
            assert "'dsp_frente_a_lo_importado' not in t" in fh.read()
