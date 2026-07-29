"""
Tests de la CONTABILIDAD de AudD por via (source) + cap por dispositivo de
Escuchar (/recognize).

Antes solo /analyze se logueaba en audd_call_log; Escuchar (la feature estrella)
era gasto invisible e ilimitado. Ahora:
  - log_audd_call(source=...) registra las tres vias.
  - count_audd_calls_today (cap de /analyze) EXCLUYE recognize/identify.
  - count_recognitions_today(device) cuenta Escuchar por dispositivo/dia.
"""

import os
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'acct.db'))


def test_log_persists_source_and_device():
    db = _db()
    db.log_audd_call('recognize', True, source='recognize', device_id='devA')
    assert db.count_recognitions_today('devA') == 1


def test_analyze_cap_excludes_recognize_and_identify():
    db = _db()
    # 2 de analyze (una explicita, una legacy con source por defecto).
    db.log_audd_call('fp1', True, source='analyze')
    db.log_audd_call('fp2', False)  # default source='analyze'
    # Ruido de otras vias que NO debe contar para el cap de /analyze.
    db.log_audd_call('recognize', True, source='recognize', device_id='devA')
    db.log_audd_call('recognize', False, source='recognize', device_id='devB')
    db.log_audd_call('fp3', True, source='identify')
    assert db.count_audd_calls_today() == 2


def test_legacy_null_source_counts_as_analyze():
    """Filas viejas (source NULL) deben seguir contando para el cap de /analyze."""
    db = _db()
    # Insercion cruda simulando una fila legacy sin columna source poblada.
    conn = db._open_conn()
    import time as _t
    conn.execute(
        'INSERT INTO audd_call_log (fingerprint, called_at, success) VALUES (?,?,?)',
        ('legacy', _t.time(), 1))
    conn.commit()
    conn.close()
    assert db.count_audd_calls_today() == 1


def test_recognitions_per_device_isolated():
    db = _db()
    db.log_audd_call('recognize', True, source='recognize', device_id='devA')
    db.log_audd_call('recognize', True, source='recognize', device_id='devA')
    db.log_audd_call('recognize', True, source='recognize', device_id='devB')
    assert db.count_recognitions_today('devA') == 2
    assert db.count_recognitions_today('devB') == 1
    assert db.count_recognitions_today('devC') == 0


def test_recognitions_none_device_is_zero():
    db = _db()
    db.log_audd_call('recognize', True, source='recognize', device_id=None)
    # Sin device no se puede atribuir -> no se capa (cuenta 0 para cualquiera).
    assert db.count_recognitions_today(None) == 0
    assert db.count_recognitions_today('') == 0


def test_recognize_calls_do_not_touch_analyze_cap():
    """Regresion: una racha de Escuchar NO debe agotar el cap de /analyze."""
    db = _db()
    for _ in range(50):
        db.log_audd_call('recognize', True, source='recognize', device_id='devA')
    assert db.count_audd_calls_today() == 0
    assert db.count_recognitions_today('devA') == 50
