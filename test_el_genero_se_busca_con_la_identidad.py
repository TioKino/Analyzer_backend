"""El género se busca con la identidad YA resuelta, y de ESE tema (2026-10-06).

Hasta ese día, `/analyze` consultaba Discogs y MusicBrainz con el artista y el
título de las ETIQUETAS y antes de nada más:

* un tema sin etiquetas, identificado por el NOMBRE del fichero o heredando la
  identidad de su cluster, no los consultaba nunca;
* uno con etiquetas basura («Unknown Artist - Track 01») los consultaba CON la
  basura, y Discogs devolvía el primer disco que encontrara;
* AudD volvía a lanzarlos tras identificar el tema, salvo si el fichero traía
  género en la etiqueta, que ya lo había rellenado antes;
* y una etiqueta basura tapaba un nombre de fichero bueno: el tema iba a AudD.

Y la búsqueda no verificaba nada: Discogs se quedaba con el PRIMER resultado y
MusicBrainz con la primera grabación, fueran de quien fueran.

    pytest test_el_genero_se_busca_con_la_identidad.py -v
"""

import os
import tempfile
import uuid

import numpy as np
import pytest
import soundfile

import audd_helper
import main
from genre_detection import MAX_FICHAS_DISCOGS, GenreDetector


# ------------------------------------------------------------------ dobles

class _Detector:
    """GenreDetector de mentira: apunta con qué se le pregunta."""

    def __init__(self, discogs=None, mb=None):
        self.discogs, self.mb, self.preguntas = discogs or {}, mb or {}, []

    def get_discogs_genre(self, artist, title):
        self.preguntas.append(('discogs', artist, title))
        g = self.discogs.get((artist, title))
        return {'genre': g, 'label': 'Drumcode', 'year': 2004} if g else None

    def get_musicbrainz_info(self, artist, title):
        self.preguntas.append(('mb', artist, title))
        g = self.mb.get((artist, title))
        return {'genre': g} if g else None


@pytest.fixture
def detector(monkeypatch):
    def poner(**kw):
        d = _Detector(**kw)
        monkeypatch.setattr(main, 'genre_detector', d)
        monkeypatch.setattr(main, 'GENRE_DETECTOR_ENABLED', True)
        return d
    return poner


@pytest.fixture
def sin_audd(monkeypatch):
    monkeypatch.setattr(main, 'AUDD_AUTO_ENABLED', False)


def _identidad(id3, nombre, **kw):
    return main._identidad_y_genero('/tmp/tmpabc123.mp3', uuid.uuid4().hex,
                                    400.0, id3, nombre, **kw)


# ------------------------------------------------------- la identidad

def test_las_etiquetas_buenas_mandan():
    assert main._resolver_identidad(
        {'artist': 'Adam Beyer', 'title': 'Your Mind'}, '/tmp/x.mp3',
        'Otro - Nombre.mp3') == ('Adam Beyer', 'Your Mind')


def test_lo_que_falta_sale_del_nombre_del_fichero():
    assert main._resolver_identidad(
        {'artist': 'Adam Beyer'}, '/tmp/x.mp3',
        'Adam Beyer - Your Mind.mp3') == ('Adam Beyer', 'Your Mind')


def test_EL_CASO_una_etiqueta_basura_no_tapa_un_nombre_bueno():
    assert main._resolver_identidad(
        {'artist': 'Unknown Artist', 'title': 'Track 01'}, '/tmp/x.mp3',
        'Adam Beyer - Your Mind.mp3') == ('Adam Beyer', 'Your Mind')


def test_si_los_dos_son_basura_queda_como_estaba():
    assert main._resolver_identidad(
        {'artist': 'Unknown Artist', 'title': 'Track 01'}, '/tmp/x.mp3',
        '01 - Intro.mp3') == ('Unknown Artist', 'Track 01')


def test_sin_nombre_real_no_se_usa_el_del_temporal():
    a, t = main._resolver_identidad({}, '/tmp/tmpabc123.mp3', None)
    assert a is None, 'el temporal no dice quién es'


# ------------------------------------------------------------ el género

def test_EL_CASO_un_tema_sin_etiquetas_busca_su_genero(detector, sin_audd):
    d = detector(discogs={('Adam Beyer', 'Your Mind'): 'Techno'})
    r = _identidad({}, 'Adam Beyer - Your Mind.mp3')
    assert d.preguntas == [('discogs', 'Adam Beyer', 'Your Mind')]
    assert (r['genre'], r['genre_source']) == ('Techno', 'discogs')
    assert (r['label'], r['year']) == ('Drumcode', '2004')


def test_con_basura_no_se_pregunta_a_nadie(detector, sin_audd):
    d = detector(discogs={('Unknown Artist', 'Track 01'): 'Pop'})
    r = _identidad({'artist': 'Unknown Artist', 'title': 'Track 01',
                    'genre': 'Techno'}, '01 - Intro.mp3')
    assert d.preguntas == [], 'Discogs devolvía el disco que fuera'
    assert (r['genre'], r['genre_source']) == ('Techno', 'id3')


def test_se_busca_con_lo_que_identifica_audd(detector, monkeypatch):
    """Y aunque el fichero traiga género en la etiqueta, que antes bloqueaba
    la segunda búsqueda."""
    monkeypatch.setattr(main, 'AUDD_AUTO_ENABLED', True)
    monkeypatch.setattr(main, 'AUDD_API_TOKEN', 'x')
    monkeypatch.setattr(main, '_cluster_clean_identity', lambda *_a: None)
    monkeypatch.setattr(audd_helper, 'enrich_with_audd_if_needed',
                        lambda **_k: {'artist': 'Astrix', 'title': 'Pure Energy'})
    monkeypatch.setattr(audd_helper, 'download_artwork_from_audd',
                        lambda *_a, **_k: None)
    d = detector(discogs={('Astrix', 'Pure Energy'): 'Psy-Trance'})
    r = _identidad({'artist': 'Unknown Artist', 'title': 'Track 3',
                    'genre': 'Trance'}, 'track 3.mp3')
    assert d.preguntas == [('discogs', 'Astrix', 'Pure Energy')]
    assert (r['artist'], r['title']) == ('Astrix', 'Pure Energy')
    assert (r['genre'], r['genre_source']) == ('Psy-Trance', 'discogs')


def test_se_busca_con_la_identidad_del_cluster(detector, monkeypatch):
    monkeypatch.setattr(main, 'AUDD_AUTO_ENABLED', True)
    monkeypatch.setattr(main, 'AUDD_API_TOKEN', '')
    monkeypatch.setattr(main, '_cluster_clean_identity',
                        lambda *_a: ('Joris Voorn', 'Ringo'))
    d = detector(mb={('Joris Voorn', 'Ringo'): 'Techno'})
    r = _identidad({}, 'AUDIO_0012.mp3')
    assert d.preguntas == [('discogs', 'Joris Voorn', 'Ringo'),
                           ('mb', 'Joris Voorn', 'Ringo')]
    assert (r['genre'], r['genre_source']) == ('Techno', 'musicbrainz')


def test_sin_nada_el_genero_es_el_del_dsp(detector, sin_audd):
    detector()
    r = _identidad({}, 'Adam Beyer - Your Mind.mp3')
    assert r['genre'] is None and r['genre_source'] is None


def test_los_dos_caminos_de_analyze_pasan_por_el_mismo_sitio():
    src = open('main.py', encoding='utf-8').read()
    for nombre in ('def analyze_audio(', 'def analyze_audio_chunked('):
        i = src.index(nombre)
        cuerpo = src[i:src.index('\ndef ', i + 10)]
        assert '_identidad_y_genero(' in cuerpo, nombre
        assert 'get_discogs_genre' not in cuerpo, nombre
        assert 'get_musicbrainz_info' not in cuerpo, nombre


def test_de_punta_a_punta_en_analyze_audio(detector, monkeypatch):
    """Un WAV sin etiquetas con un nombre bueno sale con el género de Discogs."""
    monkeypatch.setattr(main, 'ARTWORK_ENABLED', False)
    monkeypatch.setattr(main, 'AUDD_AUTO_ENABLED', False)
    detector(discogs={('Adam Beyer', 'Your Mind'): 'Techno'})
    sr = 22050
    y = np.zeros(sr * 12, dtype=np.float32)
    t = np.arange(int(sr * 0.05)) / sr
    bombo = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40) * 0.8
    for i in range(0, len(y) - len(bombo), int(sr * 60 / 128)):
        y[i:i + len(bombo)] += bombo
    ruta = tempfile.mktemp(suffix='.wav')
    soundfile.write(ruta, y, sr)
    try:
        r = main.analyze_audio(ruta, fingerprint=uuid.uuid4().hex,
                               original_filename='Adam Beyer - Your Mind.wav')
    finally:
        os.remove(ruta)
    assert (r.genre, r.genre_source) == ('Techno', 'discogs')
    assert (r.artist, r.title) == ('Adam Beyer', 'Your Mind')


# ------------------------------------------- Discogs: de ESE artista y tema

class _Release:
    """Un resultado de búsqueda de discogs_client: `data` es lo de la
    búsqueda («Artista - Release») hasta que se pide la ficha."""

    def __init__(self, cabecera, ficha):
        self.data = {'title': cabecera}
        self._ficha = ficha
        self.pedidas = 0

    def refresh(self):
        self.pedidas += 1
        self.data.update(self._ficha)

    genres = property(lambda s: s.data.get('genres'))
    styles = property(lambda s: s.data.get('styles'))
    year = property(lambda s: s.data.get('year'))
    labels = property(lambda s: [])


def _pistas(*titulos, artista=None):
    return [{'title': t, **({'artists': [{'name': artista}]} if artista else {})}
            for t in titulos]


def test_EL_CASO_discogs_no_se_queda_con_el_disco_de_otro():
    otro = _Release('Jeff Mills - Your Mind', {'tracklist': _pistas('Your Mind'),
                                              'styles': ['Detroit Techno']})
    bueno = _Release('Adam Beyer - Your Mind EP',
                     {'tracklist': _pistas('Your Mind', 'Space Date'),
                      'styles': ['Techno']})
    r = GenreDetector._release_que_casa([otro, bueno], 'Adam Beyer',
                                        'Your Mind (Original Mix)')
    assert r is bueno
    assert otro.pedidas == 0, 'el artista se mira sin pedir la ficha'


def test_el_artista_bueno_sin_ese_tema_no_vale():
    ep = _Release('Adam Beyer - Decoded', {'tracklist': _pistas('Ignition Key')})
    assert GenreDetector._release_que_casa([ep], 'Adam Beyer', 'Your Mind') is None


def test_un_recopilatorio_vale_si_la_pista_es_de_ese_artista():
    va = _Release('Various - Drumcode 10', {
        'tracklist': _pistas('Your Mind', artista='Adam Beyer')})
    assert GenreDetector._release_que_casa([va], 'Adam Beyer', 'Your Mind') is va
    va_otro = _Release('Various - Techno 2004', {
        'tracklist': _pistas('Your Mind', artista='Jeff Mills')})
    assert GenreDetector._release_que_casa([va_otro], 'Adam Beyer',
                                           'Your Mind') is None


def test_un_single_sin_lista_de_pistas_vale_por_su_titulo():
    single = _Release('Adam Beyer - Your Mind', {})
    assert GenreDetector._release_que_casa([single], 'Adam Beyer',
                                           'Your Mind') is single


def test_como_mucho_tres_fichas():
    malos = [_Release('Adam Beyer - Otro %d' % i, {'tracklist': _pistas('Nada')})
             for i in range(6)]
    assert GenreDetector._release_que_casa(malos, 'Adam Beyer', 'Your Mind') is None
    assert sum(m.pedidas for m in malos) == MAX_FICHAS_DISCOGS


def test_get_discogs_genre_con_el_release_que_casa():
    class _Cliente:
        def search(self, *q, **campos):
            self.q = (q, campos)
            return [_Release('Jeff Mills - Your Mind', {'styles': ['Detroit Techno']}),
                    _Release('Adam Beyer - Your Mind', {'styles': ['Techno'],
                                                        'genres': ['Electronic'],
                                                        'year': 2004})]
    g = GenreDetector()
    g.discogs_client = _Cliente()
    r = g.get_discogs_genre('Adam Beyer', 'Your Mind (Original Mix)')
    assert r['genre'] == g._normalize_genre('Techno')
    assert r['year'] == 2004
    assert g.discogs_client.q == (('Adam Beyer Your Mind',), {'type': 'release'})
    g.discogs_client.search = lambda *a, **k: [
        _Release('Jeff Mills - Your Mind', {'styles': ['Detroit Techno']})]
    assert g.get_discogs_genre('Adam Beyer', 'Your Mind') is None


# ----------------------------------- MusicBrainz: la grabación que casa

def _grabacion(artista, titulo, tags=()):
    return {'title': titulo, 'artist-credit': [{'name': artista}],
            'tags': [{'name': t, 'count': 1} for t in tags]}


def test_EL_CASO_musicbrainz_no_se_queda_con_la_primera():
    gs = [_grabacion('Jeff Mills', 'Your Mind', ['detroit techno']),
          _grabacion('Adam Beyer', 'Your Mind'),
          _grabacion('Adam Beyer', 'Your Mind (Remix)', ['techno'])]
    g = GenreDetector._grabacion_que_casa(gs, 'Adam Beyer', 'Your Mind')
    assert g is gs[2], 'de ese artista, y mejor con etiquetas'
    assert GenreDetector._grabacion_que_casa(gs[:1], 'Adam Beyer', 'Your Mind') is None


def test_un_credito_compartido_vale():
    gs = [{'title': 'Ringo', 'artist-credit': [
        {'name': 'Joris Voorn', 'joinphrase': ' & '}, {'name': 'Kris Wadsworth'}]}]
    assert GenreDetector._grabacion_que_casa(gs, 'Joris Voorn', 'Ringo') is gs[0]
