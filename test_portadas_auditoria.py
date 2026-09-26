"""
Auditoria de portadas (2026-09-26): lo que se arreglo en el backend.

- La busqueda cogia el PRIMER resultado con imagen, sin mirar de quien era, y
  lo guardaba bajo la huella: lo veian todos los que tienen ese fichero. Y el
  ultimo recurso de Last.fm era la portada del album MAS POPULAR del artista.
- `/analyze` preferia la de internet a la del fichero si pesaba mas.
- Los AIFF/WAV no daban portada (llevan ID3 dentro y se buscaba `.pictures`).
- Cada peticion de una huella sin portada volvia a salir a internet.
- La subida pisaba siempre; ahora el cliente puede pedir `solo_si_falta`.
- El escritorio busca por su cuenta: `GET /artwork/{fp}?online=0`.
"""

import os
import struct
import tempfile
import uuid
import wave

import pytest
from fastapi.testclient import TestClient

import artwork_and_cuepoints as ac

JPG = b"\xff\xd8\xff\xe0" + b"\x05" * 20000
JPG_OTRA = b"\xff\xd8\xff\xe0" + b"\x09" * 20000
PNG = b"\x89PNG\r\n\x1a\n" + b"\x07" * 3000


class _Resp:
    def __init__(self, status=200, json_data=None, content=b""):
        self.status_code = status
        self._json = json_data
        self.content = content

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _red(monkeypatch, responder):
    pedidas = []

    def fake_get(url, timeout=None):
        pedidas.append(url)
        return responder(url)

    monkeypatch.setattr(ac.requests, "get", fake_get)
    return pedidas


class TestSoloValeLaDeEseTema:
    def test_errata_en_el_artista(self):
        assert ac.mismo_artista("Sander vand Doorn", "Sander van Doorn")
        assert ac.mismo_artista("Rricardo Villalobos", "Ricardo Villalobos")

    def test_colaboraciones(self):
        assert ac.mismo_artista("Pashka feat Ijeoma", "Pashka")
        assert ac.mismo_artista("Trentemøller", "Pashka & Trentemøller")

    def test_otro_artista_no_cuela(self):
        assert not ac.mismo_artista("Sander van Doorn", "Armin van Buuren")
        assert not ac.mismo_artista("DJ", "DJ Tiësto")

    def test_titulo_pelado(self):
        assert ac.mismo_titulo("Grasshopper", "Grasshopper (Original Mix)")
        assert not ac.mismo_titulo("Grasshopper", "Riff")

    def test_variantes_limpian_el_nombre(self):
        a, t, consultas = ac.variantes_de_busqueda("", "polder_-_cucumber")
        assert (a, t) == ("polder", "cucumber")
        _, t, consultas = ac.variantes_de_busqueda(
            "Pier Bucci", "Hay Consuelo (Samim Remix) - minimal")
        assert t == "Hay Consuelo"
        assert consultas[-1][0] is None, "la ultima variante va solo por titulo"


class TestLaBusqueda:
    def test_salta_el_de_otro_artista(self, monkeypatch):
        def responder(url):
            if "api.deezer.com" in url:
                return _Resp(json_data={"data": [
                    {"title": "Grasshopper", "artist": {"name": "Armin van Buuren"},
                     "album": {"cover_xl": "https://img/mala.jpg"}},
                    {"title": "Grasshopper (Original Mix)",
                     "artist": {"name": "Sander van Doorn"},
                     "album": {"cover_xl": "https://img/buena.jpg"}},
                ]})
            if url.endswith("mala.jpg"):
                return _Resp(content=JPG_OTRA)
            if url.endswith("buena.jpg"):
                return _Resp(content=JPG)
            return _Resp(404)

        _red(monkeypatch, responder)
        portada, definitivo = ac.buscar_portada("Sander vand Doorn", "Grasshopper")
        assert portada["data"] == JPG
        assert definitivo

    def test_nadie_la_tiene_es_definitivo(self, monkeypatch):
        def responder(url):
            if "api.deezer.com" in url:
                return _Resp(json_data={"data": []})
            if "itunes" in url:
                return _Resp(json_data={"resultCount": 0, "results": []})
            return _Resp(json_data={"error": 6, "message": "Track not found"})

        pedidas = _red(monkeypatch, responder)
        assert ac.buscar_portada("Nadie", "Nada") == (None, True)
        assert not any("getTopAlbums" in u for u in pedidas), (
            "el album mas popular del artista casi nunca es la portada del tema")

    def test_un_403_no_es_un_no(self, monkeypatch):
        def responder(url):
            if "api.deezer.com" in url:
                return _Resp(json_data={"error": {"code": 4, "message": "Quota"}})
            return _Resp(403)

        _red(monkeypatch, responder)
        assert ac.buscar_portada("Nadie", "Nada") == (None, False)


class TestQuienGanaAlAnalizar:
    def _con(self, monkeypatch, embebida, online):
        monkeypatch.setattr(ac, "extract_artwork_from_file", lambda p: embebida)
        monkeypatch.setattr(ac, "search_artwork_online", lambda a, t, al=None: online)

    def test_la_del_fichero_gana_aunque_la_de_internet_pese_mas(self, monkeypatch):
        emb = {"data": JPG[:15000], "mime_type": "image/jpeg", "size": 15000}
        onl = {"data": JPG, "mime_type": "image/jpeg", "size": 20004, "source": "deezer"}
        self._con(monkeypatch, emb, onl)
        portada, embebida, fuente = ac.elegir_portada("x.mp3", "A", "T")
        assert portada is emb and embebida and fuente == "id3"

    def test_la_de_audd_si_puede_mejorarla(self, monkeypatch):
        emb = {"data": JPG[:15000], "mime_type": "image/jpeg", "size": 15000}
        audd = {"data": JPG, "mime_type": "image/jpeg", "size": 20004, "source": "audd"}
        self._con(monkeypatch, emb, None)
        portada, embebida, _ = ac.elegir_portada("x.mp3", "A", "T", audd)
        assert portada is audd and not embebida

    def test_miniatura_del_fichero_y_nada_en_internet(self, monkeypatch):
        mini = {"data": JPG[:3000], "mime_type": "image/jpeg", "size": 3000}
        self._con(monkeypatch, mini, None)
        assert ac.elegir_portada("x.mp3", "A", "T")[0] is mini


class TestLoQueHayDentroDelFichero:
    def _wav_con_portadas(self, fotos):
        from mutagen.id3 import APIC
        from mutagen.wave import WAVE

        ruta = tempfile.mktemp(suffix=".wav")
        with wave.open(ruta, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)
        audio = WAVE(ruta)
        audio.add_tags()
        for tipo, datos in fotos:
            # mutagen ordena por descripcion: la trasera («a…») va delante,
            # para que «la primera» y «la frontal» no coincidan.
            audio.tags.add(APIC(encoding=3, mime="image/jpeg", type=tipo,
                                desc="a trasera" if tipo != 3 else "z frontal",
                                data=datos))
        audio.save()
        return ruta

    def test_wav_con_portada(self):
        ruta = self._wav_con_portadas([(3, JPG)])
        try:
            assert ac.extract_artwork_from_file(ruta)["data"] == JPG
        finally:
            os.unlink(ruta)

    def test_manda_la_frontal(self):
        ruta = self._wav_con_portadas([(4, JPG_OTRA), (3, JPG)])
        try:
            assert ac.extract_artwork_from_file(ruta)["data"] == JPG
        finally:
            os.unlink(ruta)

    def test_png_se_guarda_como_png_y_barre_la_vieja(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ac, "ARTWORK_CACHE_DIR", str(tmp_path))
        fp = uuid.uuid4().hex
        (tmp_path / f"{fp}.jpg").write_bytes(JPG)
        assert ac.save_artwork_to_cache(fp, PNG, "image/jpeg") == f"{fp}.png"
        assert not (tmp_path / f"{fp}.jpg").exists()


@pytest.fixture(scope="module")
def app_mod():
    import main

    return main


@pytest.fixture(autouse=True)
def cupo_limpio():
    import validation

    validation.artwork_online_limiter = validation.RateLimiter(
        max_requests=validation.ARTWORK_ONLINE_MAX_PER_MIN, window_seconds=60)
    yield


def _track_en_bd(app_mod):
    fp = uuid.uuid4().hex
    app_mod.db.save_track({
        'id': fp, 'filename': f'{fp}.mp3', 'artist': 'Artista', 'title': 'Tema',
        'duration': 300.0, 'bpm': 128.0, 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time', 'fingerprint': fp,
    })
    return fp


class TestLasRutas:
    def test_online_0_no_sale_a_internet(self, app_mod, monkeypatch):
        from routes import analysis_artwork as aw
        llamadas = []
        monkeypatch.setattr(aw, "buscar_portada",
                            lambda a, t: llamadas.append(1) or (None, True))
        fp = _track_en_bd(app_mod)
        r = TestClient(app_mod.app).get(f"/artwork/{fp}?online=0")
        assert r.status_code == 404 and not llamadas

    def test_un_no_definitivo_se_recuerda_y_uno_dudoso_no(self, app_mod, monkeypatch):
        from routes import analysis_artwork as aw
        llamadas = []
        definitivo = {"v": False}
        monkeypatch.setattr(aw, "buscar_portada",
                            lambda a, t: llamadas.append(1) or (None, definitivo["v"]))
        fp = _track_en_bd(app_mod)
        cliente = TestClient(app_mod.app)
        h = {"X-Forwarded-For": "203.0.113.120"}
        cliente.get(f"/artwork/{fp}", headers=h)
        cliente.get(f"/artwork/{fp}", headers=h)
        assert len(llamadas) == 2, "un «no contesto» se vuelve a intentar"
        definitivo["v"] = True
        cliente.get(f"/artwork/{fp}", headers=h)
        cliente.get(f"/artwork/{fp}", headers=h)
        assert len(llamadas) == 3, "un «no hay» seguro no vuelve a salir a internet"

    def test_solo_si_falta_no_pisa(self, app_mod):
        from routes import analysis_artwork as aw
        fp = uuid.uuid4().hex
        cliente = TestClient(app_mod.app)
        r = cliente.post(f"/artwork/upload/{fp}", files={"file": ("a.jpg", JPG)})
        assert r.json()["status"] == "ok"
        try:
            r = cliente.post(f"/artwork/upload/{fp}?solo_si_falta=1",
                             files={"file": ("b.jpg", JPG_OTRA)})
            assert r.json()["status"] == "exists"
            assert cliente.get(f"/artwork/{fp}").content == JPG
            # Sin la marca, pisa (la del propio fichero).
            cliente.post(f"/artwork/upload/{fp}", files={"file": ("b.jpg", JPG_OTRA)})
            assert cliente.get(f"/artwork/{fp}").content == JPG_OTRA
        finally:
            for ext in ("jpg", "png"):
                ruta = os.path.join(aw.ARTWORK_CACHE_DIR, f"{fp}.{ext}")
                if os.path.exists(ruta):
                    os.unlink(ruta)

    def test_id_raro_es_404(self, app_mod):
        cliente = TestClient(app_mod.app)
        assert cliente.get("/artwork/x.y").status_code == 404
        assert cliente.head("/artwork/a.b").status_code == 404
