"""
Tests de la CONTABILIDAD de AudD por via (source) + cap por dispositivo de
Escuchar (/recognize).

Dos metricas distintas sobre audd_call_log:
  - COSTE real: una fila por LLAMADA AudD (source='recognize' en Escuchar,
    'analyze' en el auto-trigger, 'identify' en /identify).
  - CAP del usuario: una fila por SESION (source='recognize_session'), una por
    pulsacion de Escuchar. Se capa por sesion, no por llamada: una sesion puede
    gastar hasta 3 llamadas (3 estrategias en un fallo) y para el usuario es UN
    uso -> capar por llamada haria el cupo gratis tacaño.

Ademas: el cap de /analyze (count_audd_calls_today) NO debe verse afectado por
las llamadas de Escuchar/identify.
"""

import os
import tempfile
import time

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'acct.db'))


def test_log_persists_source_and_device():
    db = _db()
    db.log_audd_call('recognize_session', True,
                     source='recognize_session', device_id='devA')
    assert db.count_recognition_sessions_today('devA') == 1


def test_analyze_cap_excludes_other_sources():
    db = _db()
    # 2 de analyze (una explicita, una legacy con source por defecto).
    db.log_audd_call('fp1', True, source='analyze')
    db.log_audd_call('fp2', False)  # default source='analyze'
    # Ruido de otras vias que NO debe contar para el cap de /analyze.
    db.log_audd_call('recognize', True, source='recognize', device_id='devA')
    db.log_audd_call('recognize_session', True,
                     source='recognize_session', device_id='devA')
    db.log_audd_call('fp3', True, source='identify')
    assert db.count_audd_calls_today() == 2


def test_legacy_null_source_counts_as_analyze():
    db = _db()
    conn = db._open_conn()
    conn.execute(
        'INSERT INTO audd_call_log (fingerprint, called_at, success) VALUES (?,?,?)',
        ('legacy', time.time(), 1))
    conn.commit()
    conn.close()
    assert db.count_audd_calls_today() == 1


def test_sessions_per_device_isolated():
    db = _db()
    db.log_audd_call('recognize_session', True,
                     source='recognize_session', device_id='devA')
    db.log_audd_call('recognize_session', False,
                     source='recognize_session', device_id='devA')
    db.log_audd_call('recognize_session', True,
                     source='recognize_session', device_id='devB')
    assert db.count_recognition_sessions_today('devA') == 2
    assert db.count_recognition_sessions_today('devB') == 1
    assert db.count_recognition_sessions_today('devC') == 0


def test_sessions_none_device_is_zero():
    db = _db()
    db.log_audd_call('recognize_session', True,
                     source='recognize_session', device_id=None)
    assert db.count_recognition_sessions_today(None) == 0
    assert db.count_recognition_sessions_today('') == 0


def test_cap_counts_sessions_not_calls():
    """3 llamadas (un fallo que probo 3 estrategias) = 1 sesion para el cap."""
    db = _db()
    for _ in range(3):  # 3 llamadas AudD de la MISMA sesion (fallo)
        db.log_audd_call('recognize', False,
                         source='recognize', device_id='devA')
    db.log_audd_call('recognize_session', False,  # 1 marcador de sesion
                     source='recognize_session', device_id='devA')
    assert db.count_recognition_sessions_today('devA') == 1  # el usuario: 1 uso


def test_recognize_does_not_touch_analyze_cap():
    """Regresion: una racha de Escuchar NO agota el cap de /analyze."""
    db = _db()
    for _ in range(50):
        db.log_audd_call('recognize', True,
                         source='recognize', device_id='devA')
        db.log_audd_call('recognize_session', True,
                         source='recognize_session', device_id='devA')
    assert db.count_audd_calls_today() == 0
    assert db.count_recognition_sessions_today('devA') == 50
