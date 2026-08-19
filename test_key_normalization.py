"""Normalizacion de tonalidad — que la key de Rekordbox no se pierda.

Regresion de 2026-08-19: `/analyze` solo aceptaba el TKEY si coincidia LETRA A
LETRA con una clave de KEY_TO_CAMELOT (que solo habla en sostenidos). Rekordbox,
Traktor y Mixed In Key escriben en bemoles ('Bbm', 'Ab'), asi que la tonalidad
que el usuario tenia currada se descartaba en silencio y ganaba nuestro DSP.
"""
import pytest

from main import KEY_TO_CAMELOT, normalize_musical_key


class TestEnharmonics:
    """Bemoles: la notacion por defecto de Rekordbox."""

    @pytest.mark.parametrize('raw,expected', [
        ('Bbm', ('A#m', '3A')),
        ('Abm', ('G#m', '1A')),
        ('Ebm', ('D#m', '2A')),
        ('Dbm', ('C#m', '12A')),
        ('Gbm', ('F#m', '11A')),
        ('Ab', ('G#', '4B')),
        ('Bb', ('A#', '6B')),
        ('Db', ('C#', '3B')),
        ('Eb', ('D#', '5B')),
        ('Gb', ('F#', '2B')),
    ])
    def test_flats_map_to_sharps(self, raw, expected):
        assert normalize_musical_key(raw) == expected

    def test_b_natural_is_not_read_as_a_flat(self):
        # 'B' es la nota si, no un bemol suelto. Confundirlas mandaria 1B a 6B.
        assert normalize_musical_key('B') == ('B', '1B')
        assert normalize_musical_key('Bm') == ('Bm', '10A')


class TestModeSpellings:
    @pytest.mark.parametrize('raw', ['Fm', 'FM', 'fm', 'F min', 'F minor',
                                     'Fmin', 'F-min', 'F_MINOR', 'f moll'])
    def test_minor_spellings(self, raw):
        assert normalize_musical_key(raw) == ('Fm', '4A')

    @pytest.mark.parametrize('raw', ['C', 'c', 'C maj', 'Cmaj', 'C major', 'C dur'])
    def test_major_spellings(self, raw):
        assert normalize_musical_key(raw) == ('C', '8B')

    def test_unicode_sharp_and_flat(self):
        assert normalize_musical_key('F♯m') == ('F#m', '11A')
        assert normalize_musical_key('B♭m') == ('A#m', '3A')


class TestCamelotInput:
    """Una key guardada como codigo Camelot debe volver a nota musical.

    Si no, la columna KEY de la biblioteca acaba mostrando '4A' en vez de 'Fm'
    y parece que key y camelot 'no se corresponden'.
    """

    @pytest.mark.parametrize('raw,expected', [
        ('8A', ('Am', '8A')),
        ('8a', ('Am', '8A')),
        ('08A', ('Am', '8A')),
        ('12B', ('E', '12B')),
        ('4 A', ('Fm', '4A')),
    ])
    def test_camelot_roundtrip(self, raw, expected):
        assert normalize_musical_key(raw) == expected

    def test_open_key_traktor(self):
        # Open Key 1d = Do mayor = 8B; 1m = La menor = 8A.
        assert normalize_musical_key('1d') == ('C', '8B')
        assert normalize_musical_key('1m') == ('Am', '8A')
        assert normalize_musical_key('6m') == ('G#m', '1A')


class TestRejects:
    @pytest.mark.parametrize('raw', [None, '', '   ', '?', 'Unknown', 'None',
                                     '-', 'H', 'Zm', '13A', '0A', 42, 'techno'])
    def test_garbage_is_rejected(self, raw):
        assert normalize_musical_key(raw) is None


class TestInvariant:
    def test_every_canonical_key_survives_roundtrip(self):
        for key, camelot in KEY_TO_CAMELOT.items():
            assert normalize_musical_key(key) == (key, camelot)

    def test_result_always_satisfies_key_to_camelot(self):
        for raw in ['Bbm', 'Ab', '8A', 'F minor', 'c#M', '1m']:
            key, camelot = normalize_musical_key(raw)
            assert KEY_TO_CAMELOT[key] == camelot
