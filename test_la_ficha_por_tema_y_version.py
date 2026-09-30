"""La ficha por nombre: el mismo TEMA y la misma VERSION (2026-09-30).

En la primera prueba de Shazam en el iPhone del owner, «Age of Love – The Age
Of Love (Jam & Spoon Watch Out For Stella Mix)» salió dos veces sin ficha de
la comunidad con el tema analizado: Shazam escribe el remix entre paréntesis,
un tag lo escribe tras un guion, y `LIKE '%titulo%'` sobre el título casi
entero no casaba. Y al revés, el título sin versión casaba con cualquier
remix: la ficha enseñaba el BPM y la tonalidad de otra versión.

Lo que se ata:
- Paréntesis, corchetes o guion; `&` o `and`; acentos o no: el mismo tema.
- «Original Mix», «Extended Mix», «Radio Edit»… son el original.
- Un remix NUNCA coge la ficha de otra versión (ni del original).
- El artista tolera colaboraciones.
- Cada acierto de Escuchar apunta si salió con ficha (`con_ficha`).

    pytest test_la_ficha_por_tema_y_version.py -v
"""

import uuid

import pytest
from fastapi.testclient import TestClient

import main
from routes.search import buscar_analizado, tema_y_version


def _guardar(artist, title, bpm=128.0, analyzed_at='2025-06-01T00:00:00'):
    fp = uuid.uuid4().hex
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3',
        'artist': artist, 'title': title, 'bpm': bpm, 'duration': 300,
        'key': 'Am', 'camelot': '8A', 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time', 'analyzed_at': analyzed_at,
    })
    return fp


def _s():
    return uuid.uuid4().hex[:6]


class TestTemaYVersion:
    @pytest.mark.parametrize('a,b', [
        ('The Age Of Love (Jam & Spoon Watch Out For Stella Mix)',
         'The Age of Love - Jam and Spoon Watch Out for Stella Mix'),
        ('The Age of Love (Charlotte De Witte & Enrico Sanguiliano Remix)',
         'The Age Of Love [Charlotte de Witte and Enrico Sanguiliano Mix]'),
        ('Rave', 'Rave (Original Mix)'),
        ('Rave', 'Rave (Extended Mix)'),
        ('Rave (Radio Edit)', 'Rave'),
        ('Adagio for Strings (feat. X)', 'Adagio For Strings'),
        ('Café del Mar', 'Cafe Del Mar'),
    ])
    def test_lo_mismo_escrito_distinto(self, a, b):
        assert tema_y_version(a) == tema_y_version(b)

    @pytest.mark.parametrize('a,b', [
        ('Rave', 'Rave (Adam Beyer Remix)'),
        ('Rave (Remix)', 'Rave'),
        ('The Age of Love (Jam & Spoon Watch Out For Stella Mix)',
         'The Age of Love (Charlotte De Witte & Enrico Sanguiliano Remix)'),
        ('Rave (Dub Mix)', 'Rave (Club Mix)'),
    ])
    def test_otra_version_no_es_la_misma(self, a, b):
        assert tema_y_version(a) != tema_y_version(b)

    def test_un_guion_que_no_es_version_es_parte_del_titulo(self):
        assert tema_y_version('Love - Tomorrow')[1] == frozenset()


class TestBuscar:
    def test_el_caso_de_age_of_love(self):
        s = _s()
        fp = _guardar(f'Age Of Love {s}',
                      f'The Age of Love {s} - Jam and Spoon Watch Out for Stella Mix')
        otro = _guardar(f'Age Of Love {s}', f'The Age of Love {s}',
                        analyzed_at='2026-01-01')
        f = buscar_analizado(
            f'Age of Love {s}',
            f'The Age Of Love {s} (Jam & Spoon Watch Out For Stella Mix)')
        assert f and f['id'] == fp, 'la del remix, no la del original'
        assert buscar_analizado(f'Age of Love {s}',
                                f'The Age of Love {s} (Original Mix)')['id'] == otro

    def test_un_remix_no_coge_la_ficha_del_original(self):
        s = _s()
        _guardar(f'Sam Paganini {s}', f'Rave {s}')
        assert buscar_analizado(f'Sam Paganini {s}',
                                f'Rave {s} (Adam Beyer Remix)') is None

    def test_ni_el_original_la_de_un_remix(self):
        s = _s()
        _guardar(f'Sam Paganini {s}', f'Rave {s} (Adam Beyer Remix)')
        assert buscar_analizado(f'Sam Paganini {s}', f'Rave {s}') is None

    def test_el_artista_tolera_colaboraciones(self):
        s = _s()
        fp = _guardar(f'Age Of Love {s}, Charlotte de Witte, Enrico Sanguiliano',
                      f'The Age Of Love {s} (Charlotte de Witte & Enrico Sanguiliano Remix)')
        f = buscar_analizado(
            f'Age of Love {s}',
            f'The Age of Love {s} (Charlotte De Witte & Enrico Sanguiliano Remix)')
        assert f and f['id'] == fp

    def test_acentos(self):
        s = _s()
        fp = _guardar(f'Tiësto {s}', f'Adagio for Strings {s}')
        assert buscar_analizado(f'Tiesto {s}', f'Adagio For Strings {s}')['id'] == fp
        fp2 = _guardar(f'Jose Padilla {s}', f'Café del Mar {s}')
        assert buscar_analizado(f'José Padilla {s}', f'Cafe Del Mar {s}')['id'] == fp2

    def test_un_numero_de_pista_delante(self):
        s = _s()
        fp = _guardar(f'Adam Beyer {s}', f'01 Your Mind {s}')
        assert buscar_analizado(f'Adam Beyer {s}', f'Your Mind {s}')['id'] == fp

    def test_gana_la_misma_version_exacta_sobre_la_contenida(self):
        s = _s()
        _guardar(f'Adam Beyer {s}', f'01 Your Mind {s}', analyzed_at='2026-05-01')
        exacto = _guardar(f'Adam Beyer {s}', f'Your Mind {s} (Original Mix)',
                          analyzed_at='2025-01-01')
        assert buscar_analizado(f'Adam Beyer {s}', f'Your Mind {s}')['id'] == exacto

    def test_otro_artista_no(self):
        s = _s()
        _guardar(f'Otro Artista {s}', f'Rave {s}')
        assert buscar_analizado(f'Sam Paganini {s}', f'Rave {s}') is None

    def test_sin_bpm_no_es_ficha(self):
        s = _s()
        _guardar(f'Age Of Love {s}', f'The Age of Love {s} (Jam & Spoon Mix)',
                 bpm=0)
        assert buscar_analizado(f'Age of Love {s}',
                                f'The Age of Love {s} - Jam and Spoon Mix') is None


class TestSeMide:
    @pytest.fixture(autouse=True)
    def _entorno(self, monkeypatch):
        for v in ('ESCUCHAR_PRIMER_CLIP_S', 'ESCUCHAR_ENVIO', 'ESCUCHAR_MOTOR'):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setattr(main, 'check_rate_limit', lambda ip: None)
        monkeypatch.setattr(main, 'search_artwork_online', None)

    def _marcadores(self, dev):
        conn = main.db._open_conn()
        try:
            return [f[0] for f in conn.execute(
                "SELECT con_ficha FROM audd_call_log WHERE device_id = ? "
                "AND source = 'recognize_session' ORDER BY id", (dev,))]
        finally:
            conn.close()

    def test_cada_acierto_apunta_si_salio_con_ficha(self):
        client = TestClient(main.app)
        s = _s()
        _guardar(f'Age Of Love {s}', f'The Age of Love {s}')
        dev = f'm-{uuid.uuid4().hex[:8]}'
        antes = main.db.resumen_escuchar()['por_variante'].get(
            'shazam+12s+ffmpeg', {}).get('con_ficha', 0)
        for titulo in (f'The Age of Love {s} (Original Mix)', f'Nada {s}'):
            r = client.post('/recognize/shazam', json={
                'artist': f'Age of Love {s}', 'title': titulo,
                'origen': 'escuchar', 'sesion_id': f's{_s()}',
                'device_id': dev})
            assert r.json()['status'] == 'found'
        assert self._marcadores(dev) == [1, 0]
        despues = main.db.resumen_escuchar()['por_variante'][
            'shazam+12s+ffmpeg']['con_ficha']
        assert despues - antes == 1

    def test_tambien_con_audd(self, monkeypatch):
        client = TestClient(main.app)
        s = _s()
        _guardar(f'Sam Paganini {s}', f'Rave {s}')
        monkeypatch.setattr(main, '_preprocess_audio_for_recognition',
                            lambda e, sal, est: open(sal, 'wb').write(b'\0' * 4096) or True)
        monkeypatch.setattr(main, '_send_to_audd', lambda p, t, timeout=30: (
            {'artist': f'Sam Paganini {s}', 'title': f'Rave {s} (Extended Mix)'},
            True))
        dev = f'm-{uuid.uuid4().hex[:8]}'
        r = client.post('/recognize', files={'file': ('a.m4a', b'x' * 5000)},
                        data={'device_id': dev, 'origen': 'escuchar',
                              'sesion_id': 's1'})
        assert r.json()['backend_analysis'] is not None
        assert self._marcadores(dev) == [1]
