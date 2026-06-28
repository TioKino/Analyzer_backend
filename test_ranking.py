"""Tests unitarios para analysis_ranking — logica pura sin librosa.

Cubre el item 8 del PENDING.md: "analisis mejor gana en sync comunitario".
"""

from analysis_ranking import (
    get_source_priority,
    should_overwrite_analysis,
)


class TestSourcePriority:
    """get_source_priority ranquea las fuentes que pueden escribir un track
    en la cache comunitaria. Rekordbox/Traktor profesional > consensus
    comunitario fuerte > APIs > motor local > metadata > analisis spectral."""

    def test_rekordbox_beats_local_engine(self):
        assert get_source_priority('rekordbox') > get_source_priority('local_engine')

    def test_traktor_beats_local_engine(self):
        assert get_source_priority('traktor') > get_source_priority('local_engine')

    def test_local_engine_beats_id3(self):
        assert get_source_priority('local_engine') > get_source_priority('id3')

    def test_id3_beats_generic_analysis(self):
        assert get_source_priority('id3') > get_source_priority('analysis')

    def test_consensus_grows_with_votes(self):
        assert get_source_priority('consensus_5') > get_source_priority('consensus_3')
        assert get_source_priority('consensus_8') > get_source_priority('consensus_5')

    def test_consensus_5_beats_local_engine(self):
        assert get_source_priority('consensus_5') > get_source_priority('local_engine')

    def test_unknown_source_priority_zero(self):
        assert get_source_priority('foobar') == 0
        assert get_source_priority(None) == 0
        assert get_source_priority('') == 0

    def test_consensus_dynamic_n(self):
        # consensus_99 sigue siendo high-prio aunque no este tabulado
        assert get_source_priority('consensus_99') >= get_source_priority('consensus_5')


class TestShouldOverwriteAnalysis:
    """should_overwrite_analysis decide si un upload a /cache-analysis
    debe machacar al existente. Es la regla central del item 8."""

    def test_no_existing_always_overwrites(self):
        assert should_overwrite_analysis(None, {'bpm_source': 'analysis'}) is True
        assert should_overwrite_analysis({}, {'bpm_source': 'analysis'}) is True

    def test_higher_priority_overwrites(self):
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "local_engine"}'}
        new = {'bpm_source': 'rekordbox', 'bpm': 128.0}
        assert should_overwrite_analysis(existing, new) is True

    def test_lower_priority_skips(self):
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "rekordbox"}'}
        new = {'bpm_source': 'analysis', 'bpm': 128.0}
        assert should_overwrite_analysis(existing, new) is False

    def test_same_priority_keeps_first(self):
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "local_engine"}'}
        new = {'bpm_source': 'local_engine', 'bpm': 130.0}
        assert should_overwrite_analysis(existing, new) is False

    def test_same_priority_but_existing_empty_lets_new_in(self):
        # Existente tiene la fuente correcta pero datos vacios (bpm=0)
        existing = {'analysis_json': '{"bpm": 0, "key": "", "bpm_source": "local_engine"}'}
        new = {'bpm_source': 'local_engine', 'bpm': 128.0, 'key': 'Am'}
        assert should_overwrite_analysis(existing, new) is True

    def test_consensus_5_beats_local_engine(self):
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "local_engine"}'}
        new = {'bpm_source': 'consensus_5', 'bpm': 130.0}
        assert should_overwrite_analysis(existing, new) is True

    def test_rekordbox_not_overwritten_by_consensus_3(self):
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "rekordbox"}'}
        new = {'bpm_source': 'consensus_3', 'bpm': 130.0}
        assert should_overwrite_analysis(existing, new) is False

    def test_pending_existing_lets_new_in(self):
        existing = {'analysis_json': '{"bpm": 0, "key": null, "bpm_source": "pending"}'}
        new = {'bpm_source': 'analysis', 'bpm': 128.0, 'key': 'Am'}
        assert should_overwrite_analysis(existing, new) is True

    def test_unknown_new_source_does_not_overwrite_known(self):
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "id3"}'}
        new = {'bpm_source': 'unknown_provider', 'bpm': 130.0}
        assert should_overwrite_analysis(existing, new) is False

    def test_flat_existing_no_analysis_json_uses_top_level(self):
        # Existente viene sin analysis_json (caso fallback BD): los campos
        # bpm/key/bpm_source viven en el dict raiz.
        existing = {'bpm': 128, 'key': 'Am', 'bpm_source': 'local_engine'}
        new = {'bpm_source': 'rekordbox', 'bpm': 128.0}
        assert should_overwrite_analysis(existing, new) is True

    def test_high_priority_empty_does_not_overwrite_valid(self):
        # Salvaguarda anti-pérdida: un import de alta prioridad SIN bpm (track
        # sin BPM en el XML de Rekordbox) NO debe machacar un análisis válido.
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "local_engine"}'}
        new = {'bpm_source': 'rekordbox', 'bpm': 0}
        assert should_overwrite_analysis(existing, new) is False

    def test_high_priority_without_key_still_overwrites(self):
        # Un análisis válido (bpm>0) pero SIN key NO es "vacío" — la key vacía
        # es normal (baja confianza) y no debe bloquear el overwrite por prioridad.
        existing = {'analysis_json': '{"bpm": 128, "key": "Am", "bpm_source": "local_engine"}'}
        new = {'bpm_source': 'rekordbox', 'bpm': 124.0}
        assert should_overwrite_analysis(existing, new) is True
