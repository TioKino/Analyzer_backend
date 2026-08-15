"""Desglose del POR QUE fallan las sesiones de Escuchar.

El panel reportaba 61 exitos de 142 llamadas de /recognize y no habia forma de
saber si ese ~57% de fallos era:

  - 'no_match'       → AudD proceso el audio y no conoce el track. Techo de SU
                       catalogo (musica underground/promo). Ni el usuario ni
                       nosotros podemos hacer nada; reintentar es tirar cuota.
  - 'audio_unusable' → AudD no pudo generar huella (ruido/silencio/audio corto).
                       Accionable: acercar el micro, mejorar la captura.

/recognize ya distinguia los dos casos para elegir el mensaje al usuario, pero
el dato se tiraba. Ahora se sella en audd_call_log.reason y se agrega por
SESION (una pulsacion de Escuchar), no por llamada.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import AnalysisDB  # noqa: E402
from main import _recognize_reason  # noqa: E402


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    d = AnalysisDB(path)
    yield d
    try:
        os.unlink(path)
    except OSError:
        pass


class TestRecognizeReason:
    """La funcion pura que decide el desenlace."""

    def test_con_match_es_matched(self):
        assert _recognize_reason({'artist': 'X'}, True) == 'matched'

    def test_match_manda_aunque_no_conste_procesado(self):
        """Si hay track_data, hubo match: el flag de procesado es irrelevante."""
        assert _recognize_reason({'artist': 'X'}, False) == 'matched'

    def test_audio_procesado_sin_match_es_no_match(self):
        assert _recognize_reason(None, True) == 'no_match'

    def test_audio_no_procesado_es_audio_unusable(self):
        assert _recognize_reason(None, False) == 'audio_unusable'

    def test_dict_vacio_cuenta_como_sin_match(self):
        """AudD puede devolver {} — no es un match."""
        assert _recognize_reason({}, True) == 'no_match'


class TestGetRecognizeReasons:

    def _session(self, db, reason, device_id='dev1'):
        db.log_audd_call(
            fingerprint='recognize_session',
            success=(reason == 'matched'),
            source='recognize_session', device_id=device_id, reason=reason,
        )

    def test_sin_datos_devuelve_vacio(self, db):
        assert db.get_recognize_reasons(days=30) == {}

    def test_agrupa_por_reason(self, db):
        for _ in range(3):
            self._session(db, 'matched')
        for _ in range(5):
            self._session(db, 'no_match')
        self._session(db, 'audio_unusable')

        out = db.get_recognize_reasons(days=30)
        assert out == {'matched': 3, 'no_match': 5, 'audio_unusable': 1}

    def test_solo_cuenta_sesiones_no_llamadas(self, db):
        """El coste (source='recognize', una fila por llamada) no debe
        contaminar el uso (source='recognize_session', una por pulsacion)."""
        self._session(db, 'no_match')
        for _ in range(3):
            db.log_audd_call(
                fingerprint='recognize', success=False,
                source='recognize', device_id='dev1', reason='no_match',
            )
        assert db.get_recognize_reasons(days=30) == {'no_match': 1}

    def test_ignora_otras_vias(self, db):
        self._session(db, 'matched')
        db.log_audd_call(fingerprint='fp1', success=True, source='analyze')
        db.log_audd_call(fingerprint='fp2', success=False, source='identify')
        assert db.get_recognize_reasons(days=30) == {'matched': 1}

    def test_filas_legacy_sin_reason_salen_como_unknown(self, db):
        """Sesiones anteriores al ALTER: no se puede inferir a posteriori."""
        db.log_audd_call(
            fingerprint='recognize_session', success=False,
            source='recognize_session', device_id='dev1',
        )
        self._session(db, 'no_match')
        out = db.get_recognize_reasons(days=30)
        assert out == {'unknown': 1, 'no_match': 1}

    def test_respeta_la_ventana_de_dias(self, db):
        import time as _t
        self._session(db, 'matched')
        conn = db._open_conn()
        try:
            conn.execute(
                "UPDATE audd_call_log SET called_at = ? "
                "WHERE source = 'recognize_session'",
                (_t.time() - 40 * 86400,),
            )
            conn.commit()
        finally:
            conn.close()
        assert db.get_recognize_reasons(days=30) == {}
        assert db.get_recognize_reasons(days=60) == {'matched': 1}


class TestNoRompeLaContabilidadExistente:
    """El `reason` es aditivo: los contadores de gasto no cambian."""

    def test_stats_por_via_siguen_excluyendo_el_marcador(self, db):
        db.log_audd_call(fingerprint='fp1', success=True,
                         source='recognize', reason='matched')
        db.log_audd_call(fingerprint='fp2', success=False,
                         source='recognize', reason='no_match')
        db.log_audd_call(fingerprint='recognize_session', success=False,
                         source='recognize_session', reason='no_match')

        stats = db.get_audd_stats_by_source(days=30)
        assert stats['recognize'] == {'total': 2, 'success': 1, 'fail': 1}
        assert 'recognize_session' not in stats

    def test_log_sin_reason_sigue_funcionando(self, db):
        """Las vias que no distinguen desenlace no tienen que pasar nada."""
        db.log_audd_call(fingerprint='fp1', success=True, source='analyze')
        assert db.get_audd_stats_by_source(days=30)['analyze']['total'] == 1
