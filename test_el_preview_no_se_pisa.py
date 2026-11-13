"""El preview de una huella no se pisa con un `curl` (2026-11-12, auditoría S4).

Era SEC-05 en agosto. La portada (`/artwork/upload`) se arregló el 26-sep; el
preview se quedó como estaba: sin credencial, sin mirar que fuera audio, pisando
siempre, con el fichero entero en memoria antes de mirar cuánto pesa, y servido
con un año de caché. Reproducido contra el servidor en modo producción:

    POST /preview/upload/abab…  'ESTO NO ES AUDIO…'   -> 200
    POST /preview/upload/abab…  'OTRA COSA DISTINTA…' -> 200   (pisa)

Y ni el preview ni la portada tenían margen de disco: el id es cualquier
cadena hexadecimal y `/data` es el disco de las dos bases de datos.

    pytest test_el_preview_no_se_pisa.py -v
"""
import io
import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("PREVIEWS_DIR", tempfile.mkdtemp())
    os.environ.setdefault("ARTWORK_CACHE_DIR", tempfile.mkdtemp())
    os.environ.pop("RENDER", None)
    import main
    import sync_endpoints

    sync_endpoints.SYNC_AUTH_SECRET = ""  # dev mode: sin firma
    return TestClient(main.app)


def _fp() -> str:
    return uuid.uuid4().hex


def _mp3(relleno: bytes = b"\x00", n: int = 400) -> bytes:
    """Algo que empieza como un MP3 de ffmpeg (etiqueta ID3v2)."""
    return b"ID3\x04\x00\x00\x00\x00\x00\x23" + relleno * n


def _subir(client, fp, cuerpo, cabeceras=None, params=""):
    return client.post(
        f"/preview/upload/{fp}{params}",
        files={"file": (f"{fp}.mp3", io.BytesIO(cuerpo), "audio/mpeg")},
        headers=cabeceras or {},
    )


def _token_de_un_aparato(client) -> str:
    device = f"pc-{uuid.uuid4().hex[:12]}"
    r = client.post("/sync/register", json={"device_id": device,
                                            "device_type": "windows"})
    assert r.status_code == 200, r.text
    return r.json()["device_token"]


class TestLoQueEntra:
    def test_un_mp3_entra_y_se_sirve(self, client):
        fp = _fp()
        cuerpo = _mp3()

        r = _subir(client, fp, cuerpo)

        assert r.status_code == 200
        assert r.json() == {"status": "ok", "track_id": fp, "size": len(cuerpo)}
        assert client.get(f"/preview/{fp}").content == cuerpo

    def test_una_trama_mpeg_sin_etiqueta_tambien(self, client):
        r = _subir(client, _fp(), b"\xff\xfb\x90\x64" + b"\x00" * 400)
        assert r.status_code == 200

    def test_lo_que_no_es_audio_no_entra(self, client):
        fp = _fp()

        r = _subir(client, fp, b"ESTO NO ES AUDIO " * 20)

        assert r.status_code == 400
        assert client.get(f"/preview/{fp}").status_code == 404

    def test_demasiado_pequeno_no_entra(self, client):
        assert _subir(client, _fp(), b"ID3" + b"\x00" * 10).status_code == 400

    def test_demasiado_grande_no_entra(self, client):
        fp = _fp()

        r = _subir(client, fp, _mp3(n=600_000))

        assert r.status_code == 400
        assert client.get(f"/preview/{fp}").status_code == 404

    def test_no_deja_un_temporal_al_lado(self, client):
        import routes.preview as rp

        fp = _fp()
        assert _subir(client, fp, _mp3()).status_code == 200
        assert [n for n in os.listdir(rp.PREVIEWS_DIR) if n.endswith(".tmp")] == []


class TestElPreviewDeVerdad:
    """El filtro no puede dejar fuera lo que el propio servidor genera."""

    def test_el_que_escribe_ffmpeg_entra(self, client, tmp_path):
        import shutil
        import subprocess

        if not shutil.which("ffmpeg"):
            pytest.skip("sin ffmpeg")
        salida = tmp_path / "preview.mp3"
        # Los mismos parámetros que `generate_preview_snippet`.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=8", "-ss", "1", "-t", "6",
             "-map", "0:a:0", "-vn", "-ac", "1", "-ab", "64k", "-ar", "22050",
             "-af", "afade=t=in:st=0:d=0.3,afade=t=out:st=5.5:d=0.5",
             str(salida)],
            check=True, timeout=60)
        cuerpo = salida.read_bytes()
        fp = _fp()

        r = _subir(client, fp, cuerpo)

        assert r.status_code == 200, r.text
        assert client.get(f"/preview/{fp}").content == cuerpo


class TestNoSePisaSinCredencial:
    def test_la_segunda_subida_anonima_no_toca_la_primera(self, client):
        fp = _fp()
        bueno = _mp3(b"\x01")
        assert _subir(client, fp, bueno).status_code == 200

        r = _subir(client, fp, _mp3(b"\x02"))

        assert r.status_code == 200, "el cliente lo da por subido: 200"
        assert r.json() == {"status": "exists", "track_id": fp}
        assert client.get(f"/preview/{fp}").content == bueno

    def test_un_token_inventado_no_es_credencial(self, client):
        fp = _fp()
        bueno = _mp3(b"\x01")
        assert _subir(client, fp, bueno).status_code == 200

        r = _subir(client, fp, _mp3(b"\x02"),
                   {"X-Device-Token": "un-token-que-no-es-de-nadie-0000"})

        assert r.json()["status"] == "exists"
        assert client.get(f"/preview/{fp}").content == bueno

    def test_un_aparato_registrado_si_puede_pisar(self, client):
        fp = _fp()
        assert _subir(client, fp, _mp3(b"\x01")).status_code == 200
        nuevo = _mp3(b"\x02")

        r = _subir(client, fp, nuevo,
                   {"X-Device-Token": _token_de_un_aparato(client)})

        assert r.json()["status"] == "ok"
        assert client.get(f"/preview/{fp}").content == nuevo

    def test_y_puede_pedir_que_no_se_pise(self, client):
        fp = _fp()
        bueno = _mp3(b"\x01")
        assert _subir(client, fp, bueno).status_code == 200

        r = _subir(client, fp, _mp3(b"\x02"),
                   {"X-Device-Token": _token_de_un_aparato(client)},
                   params="?solo_si_falta=1")

        assert r.json()["status"] == "exists"
        assert client.get(f"/preview/{fp}").content == bueno


class TestElMargenDeDisco:
    """Sin sitio no se guarda nada NUEVO: `/data` es también el disco de las
    bases de datos. Y es un 5xx: el cliente lo toma por «luego», no por un
    fallo del fichero (un 4xx lo daría por fallido para siempre)."""

    def test_un_preview_nuevo_sin_sitio_es_507(self, client, monkeypatch):
        import routes.preview as rp
        monkeypatch.setattr(rp, "hay_sitio_en_disco", lambda _d: False)
        fp = _fp()

        r = _subir(client, fp, _mp3())

        assert r.status_code == 507
        assert client.get(f"/preview/{fp}").status_code == 404

    def test_lo_que_ya_estaba_se_sigue_sirviendo(self, client, monkeypatch):
        import routes.preview as rp
        fp = _fp()
        bueno = _mp3()
        assert _subir(client, fp, bueno).status_code == 200
        monkeypatch.setattr(rp, "hay_sitio_en_disco", lambda _d: False)

        assert _subir(client, fp, _mp3(b"\x02")).json()["status"] == "exists"
        assert client.get(f"/preview/{fp}").content == bueno

    def test_una_portada_nueva_sin_sitio_es_507(self, client, monkeypatch):
        import routes.analysis_artwork as ra
        monkeypatch.setattr(ra, "hay_sitio_en_disco", lambda _d: False)
        fp = _fp()
        jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 400

        r = client.post(f"/artwork/upload/{fp}",
                        files={"file": (f"{fp}.jpg", io.BytesIO(jpg), "image/jpeg")})

        assert r.status_code == 507

    def test_con_sitio_la_portada_entra_como_siempre(self, client):
        fp = _fp()
        jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 400

        r = client.post(f"/artwork/upload/{fp}",
                        files={"file": (f"{fp}.jpg", io.BytesIO(jpg), "image/jpeg")})

        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestHaySitioEnDisco:
    def test_por_debajo_del_margen_no_hay(self, monkeypatch):
        import collections
        import validation

        Uso = collections.namedtuple("Uso", "total used free")
        monkeypatch.setattr(validation, "MIN_FREE_DISK_MB", 500)
        monkeypatch.setattr(validation.shutil, "disk_usage",
                            lambda _d: Uso(10**10, 10**10 - 100 * 2**20, 100 * 2**20))
        assert validation.hay_sitio_en_disco("/donde/sea") is False

        monkeypatch.setattr(validation.shutil, "disk_usage",
                            lambda _d: Uso(10**10, 10**9, 9 * 10**9))
        assert validation.hay_sitio_en_disco("/donde/sea") is True

    def test_si_no_se_puede_saber_se_deja_pasar(self, monkeypatch):
        import validation

        def _falla(_d):
            raise FileNotFoundError("no existe")

        monkeypatch.setattr(validation.shutil, "disk_usage", _falla)
        assert validation.hay_sitio_en_disco("/no/existe") is True


class TestPareceMp3:
    def test_reconoce_lo_que_escribe_ffmpeg(self):
        from routes.preview import parece_mp3

        assert parece_mp3(b"ID3\x04\x00")
        assert parece_mp3(b"\xff\xfb\x90\x64")   # MPEG-1 Layer III
        assert parece_mp3(b"\xff\xf3\x40\xc4")   # MPEG-2 Layer III (22.050 Hz)

    def test_no_se_traga_cualquier_cosa(self):
        from routes.preview import parece_mp3

        assert not parece_mp3(b"")
        assert not parece_mp3(b"ESTO NO ES AUDIO")
        assert not parece_mp3(b"<html>")
        assert not parece_mp3(b"\xff\x00\x00\x00")
        assert not parece_mp3(b"\x89PNG\r\n\x1a\n")
