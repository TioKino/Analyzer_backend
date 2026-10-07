"""La tonalidad, medida EN SOMBRA con varios perfiles (#4 de PENDING,
2026-10-07).

La lectura 54 dio que el DSP acierta la tonalidad de Rekordbox en el 45 % de
los temas, por los dos caminos. Para cambiar el algoritmo con números y no a
ciegas, /analyze guarda el croma medio del tema y el panel prueba sobre él
otros perfiles —y uno aprendido de esa misma música, con validación cruzada—
contra lo importado de la misma huella. Lo que ve el DJ no cambia.

    pytest test_la_tonalidad_en_sombra.py -v
"""

import json
import uuid
from datetime import datetime

import numpy as np
import pytest

import tonalidad as t
from database import AnalysisDB


def test_camelot_ida_y_vuelta_en_las_24():
    vistos = set()
    for n in range(1, 13):
        for letra in 'AB':
            c = f'{n}{letra}'
            assert t.camelot_de(*t.tonica_y_modo(c)) == c
            vistos.add(t.tonica_y_modo(c))
    assert len(vistos) == 24
    assert t.tonica_y_modo('8A') == (9, 'menor')   # La menor
    assert t.tonica_y_modo('8B') == (0, 'mayor')   # Do mayor
    assert t.tonica_y_modo('6A') == (7, 'menor')   # Sol menor (Grasshopper)


def test_las_clases_nuevas_salen_de_otra():
    c = t.clase_de_tonalidad
    assert c('8A', '11B') == 'paralela'   # La menor / La mayor
    assert c('11B', '8A') == 'paralela'
    assert c('8A', '3A') == 'semitono'    # La menor / La# menor
    assert c('8A', '1A') == 'semitono'    # La menor / Sol# menor
    assert c('8A', '3B') == 'otra'
    # Las de antes, igual.
    assert c('8A', '8B') == 'relativa'
    assert c('12A', '1A') == 'quinta'
    assert c('8A', '8a') == 'igual'
    assert c(None, '8A') is None
    # Y la medida del DSP cuenta con la misma regla.
    assert AnalysisDB._clase_de_tonalidad('8A', '11B') == 'paralela'


def _como_main(croma):
    """El bucle de `main.analyze_audio`: `np.corrcoef` con Krumhansl-Kessler,
    mayor antes que menor en cada tónica."""
    mayor = np.array(t.PERFILES['kk'][0]); menor = np.array(t.PERFILES['kk'][1])
    mayor, menor = mayor / mayor.sum(), menor / menor.sum()
    mejor, cual = -2, None
    for i in range(12):
        for perfil, modo in ((mayor, 'mayor'), (menor, 'menor')):
            r = np.corrcoef(croma, np.roll(perfil, i))[0, 1]
            if r > mejor:
                mejor, cual = r, (i, modo)
    return t.camelot_de(*cual)


def test_kk_en_sombra_es_lo_que_hace_hoy_analyze():
    """Si no, comparar «kk» con los demás no diría nada del algoritmo de hoy."""
    rnd = np.random.RandomState(7)
    for _ in range(300):
        croma = rnd.rand(12) ** 3
        croma /= croma.sum()
        assert t.tonalidad_desde_croma(croma, *t.PERFILES['kk']) == _como_main(croma)


def _tema_sintetico(rnd, tonica, modo, perfil):
    return np.roll(np.maximum(np.asarray(perfil) + rnd.rand(12) * 0.35, 0), tonica)


def test_el_perfil_aprendido_se_mide_sin_ver_el_tema():
    """Una música cuyo menor no es el de los libros (sin sexta, con la
    quinta muy marcada: lo que KK no espera). El perfil aprendido la acierta,
    y se mide con validación cruzada: lo aprende de una mitad y lo prueba en
    la otra."""
    rnd = np.random.RandomState(11)
    menor_raro = [1.0, 0.0, 0.1, 0.8, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.6, 0.0]
    mayor_raro = [1.0, 0.0, 0.2, 0.0, 0.8, 0.1, 0.0, 1.0, 0.0, 0.3, 0.0, 0.1]
    temas = []
    for k in range(400):
        tonica = rnd.randint(12)
        modo = 'menor' if k % 4 else 'mayor'
        perfil = menor_raro if modo == 'menor' else mayor_raro
        temas.append((uuid.uuid4().hex, _tema_sintetico(rnd, tonica, modo, perfil),
                      t.camelot_de(tonica, modo)))
    r = t.evaluar_perfiles(temas)
    assert r['temas'] == 400
    aprendido = r['perfiles']['aprendido']
    assert aprendido['comparados'] == 400, 'cada tema se prueba una vez'
    assert aprendido['igual'] / 400 > 0.9
    assert aprendido['igual'] >= r['perfiles']['kk']['igual']
    assert set(r['perfil_aprendido']) == {'mayor', 'menor'}


def test_con_un_solo_modo_no_se_aprende_nada():
    temas = [(uuid.uuid4().hex, np.ones(12) / 12, '8A') for _ in range(10)]
    r = t.evaluar_perfiles(temas)
    assert r['perfiles']['aprendido']['comparados'] == 0
    assert 'perfil_aprendido' not in r


def test_el_croma_que_se_guarda():
    assert t.croma_para_guardar(None) is None
    assert t.croma_para_guardar([0] * 12) is None
    assert t.croma_para_guardar([1] * 11) is None
    g = t.croma_para_guardar(np.arange(12, dtype=float))
    assert len(g) == 12 and abs(sum(g) - 1) < 1e-3
    assert t.croma_de_trozos([[], None]) is None
    assert t.croma_de_trozos([[1] + [0] * 11, [0, 1] + [0] * 10])[:2] == [0.5, 0.5]


# --------------------------------------------------------- en la base de datos

@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _tema(db, croma=None, key_source='analysis', camelot='8A'):
    fp = uuid.uuid4().hex
    fila = {
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3', 'bpm': 128.0,
        'bpm_source': 'analysis', 'key': 'Am', 'camelot': camelot,
        'key_source': key_source, 'duration': 300, 'energy_dj': 6,
        'genre': 'Techno', 'track_type': 'peak_time',
        'analyzed_at': datetime.utcnow().isoformat(),
    }
    if croma is not None:
        fila['croma'] = croma
    db.save_track(fila)
    return fp


def _la_menor():
    return t.croma_para_guardar(np.roll(t.PERFILES['kk'][1], 9))


def test_en_la_bd_cuentan_los_temas_con_croma_y_con_programa_de_fiar(db):
    con = _tema(db, croma=_la_menor())
    db.guardar_lo_importado('devA', [{'fingerprint': con, 'source': 'rekordbox',
                                      'key': 'Am', 'camelot': '8A'}])
    # Las etiquetas también valen: el croma es del audio, no de la fuente.
    id3 = _tema(db, croma=_la_menor(), key_source='id3')
    db.guardar_lo_importado('devA', [{'fingerprint': id3, 'source': 'virtualdj',
                                      'key': 'Am', 'camelot': '8A'}])
    sin_croma = _tema(db)
    db.guardar_lo_importado('devA', [{'fingerprint': sin_croma,
                                      'source': 'rekordbox', 'key': 'Am',
                                      'camelot': '8A'}])
    # Traktor no es referencia: su tabla estuvo mal hasta el 2026-09-26.
    traktor = _tema(db, croma=_la_menor())
    db.guardar_lo_importado('devA', [{'fingerprint': traktor, 'source': 'traktor',
                                      'key': 'Am', 'camelot': '8A'}])
    r = db.tonalidad_en_sombra()
    assert r['temas'] == 2
    assert r['perfiles']['kk'] == {'comparados': 2, 'igual': 2}


def test_sale_en_el_panel():
    import main  # noqa: F401  (monta el singleton que usa el panel)
    import routes.admin_panel as panel
    r = panel._tonalidad_en_sombra()
    assert r is not None, 'None = embudo.sh dice «el panel falló»'
    assert {'temas', 'perfiles'} <= set(r)
    with open(panel.__file__, encoding='utf-8') as fh:
        assert '"tonalidad_en_sombra": _tonalidad_en_sombra()' in fh.read()
    import os
    embudo = os.path.join(os.path.dirname(__file__), '..', 'Analyzer',
                          'scripts', 'embudo.sh')
    if os.path.exists(embudo):
        with open(embudo, encoding='utf-8') as fh:
            assert "'tonalidad_en_sombra' not in t" in fh.read()


def test_analyze_devuelve_el_croma():
    """El `AnalysisResult` que devuelve /analyze es el de main.py: un campo
    que solo esté en el de models.py revienta al asignarlo (CLAUDE.md)."""
    import main
    import models
    assert 'croma' in main.AnalysisResult.model_fields
    assert 'croma' in models.AnalysisResult.model_fields
