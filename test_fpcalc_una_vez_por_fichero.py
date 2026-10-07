"""Un análisis pasa el fichero por fpcalc UNA vez (auditoría del flujo de
análisis, tanda 3, 2026-10-06).

Con un nombre basura, /analyze sacaba la huella dos veces sobre el mismo
temporal: en `_cluster_clean_identity` (heredar la identidad del cluster antes
de pagar AudD) y en `_attach_acoustic` (guardarla). `compute_raw_chromaprint`
recuerda ahora la última huella de cada fichero —solo la que sale bien: el
fallo memoizado es `Analyzer_backend#79`— y lo distingue por ruta, tamaño y
fecha, así que un temporal nuevo con el mismo nombre no hereda nada.

    pytest test_fpcalc_una_vez_por_fichero.py -v
"""

import os

import pytest

import acoustic_fingerprint as af


@pytest.fixture
def fpcalc(monkeypatch):
    llamadas = []
    salida = {'ints': [1, 2, 3]}

    def una_pasada(binario, ruta, timeout):
        llamadas.append(ruta)
        ints = salida['ints']
        return (list(ints), None) if ints else (None, 'exit 2')

    monkeypatch.setattr(af, 'ensure_fpcalc', lambda: '/bin/fpcalc')
    monkeypatch.setattr(af, '_fpcalc_una_pasada', una_pasada)
    monkeypatch.setattr(af, '_merece_transcodificar', lambda motivo: False)
    monkeypatch.setattr(af, '_HUELLAS_RECIENTES', af.OrderedDict())
    return llamadas, salida


def test_el_mismo_fichero_no_vuelve_a_fpcalc(fpcalc, tmp_path):
    llamadas, _ = fpcalc
    f = tmp_path / 'subida.mp3'
    f.write_bytes(b'x' * 100)
    assert af.compute_raw_chromaprint(str(f), etiqueta='(pre-check AudD)') == [1, 2, 3]
    assert af.compute_raw_chromaprint(str(f), etiqueta='Artista - Tema.mp3') == [1, 2, 3]
    assert len(llamadas) == 1


def test_el_fallo_no_se_recuerda(fpcalc, tmp_path):
    llamadas, salida = fpcalc
    f = tmp_path / 'subida.mp3'
    f.write_bytes(b'x' * 100)
    salida['ints'] = None
    assert af.compute_raw_chromaprint(str(f)) is None
    salida['ints'] = [7, 8]
    assert af.compute_raw_chromaprint(str(f)) == [7, 8]
    assert len(llamadas) == 2


def test_otro_contenido_en_la_misma_ruta_es_otro_fichero(fpcalc, tmp_path):
    """Los temporales de Render se reutilizan de nombre: tamaño o fecha
    distintos son otro fichero."""
    llamadas, salida = fpcalc
    f = tmp_path / 'tmpab12cd.mp3'
    f.write_bytes(b'x' * 100)
    af.compute_raw_chromaprint(str(f))
    f.write_bytes(b'y' * 200)
    os.utime(f, ns=(1, 1))
    salida['ints'] = [9]
    assert af.compute_raw_chromaprint(str(f)) == [9]
    assert len(llamadas) == 2


def test_quien_la_recibe_no_puede_estropearla(fpcalc, tmp_path):
    f = tmp_path / 'a.mp3'
    f.write_bytes(b'x')
    primera = af.compute_raw_chromaprint(str(f))
    primera.append(99)
    assert af.compute_raw_chromaprint(str(f)) == [1, 2, 3]


def test_no_crece_sin_tope(fpcalc, tmp_path):
    for n in range(af._HUELLAS_MAX + 5):
        f = tmp_path / f'{n}.mp3'
        f.write_bytes(b'x')
        af.compute_raw_chromaprint(str(f))
    assert len(af._HUELLAS_RECIENTES) == af._HUELLAS_MAX
