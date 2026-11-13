"""El motor local pide AudD a Render, no lleva el token (2026-11-13, auditoría S3).

El token de AudD iba compilado en el EXE de Windows y en el DMG (y, hasta ese
mismo día, en TODAS las builds de la app): se sacaba con `strings` y con él se
gastaba la cuenta sin cupos, que los ponía el propio motor. Hoy un motor sin
token propio manda el trozo a `POST /audd/motor-local/identificar` con el
`X-Device-Token` del aparato, y Render lo pasa a AudD con su token y sus cupos
(por cuenta, por IP y en total, al día).

Toda llamada a AudD pasa por `audd_helper.post_a_audd`: las tres vías del motor
mandaban cada una su `requests.post` a api.audd.io.

    pytest test_el_motor_pide_audd_a_render.py -v
"""
import io
import os
import re
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_mod():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.pop("RENDER", None)
    import main
    import sync_endpoints

    sync_endpoints.SYNC_AUTH_SECRET = ""  # dev mode: sin firma
    return main


@pytest.fixture(scope="module")
def client(app_mod):
    return TestClient(app_mod.app)


class _Resp:
    def __init__(self, status=200, cuerpo=None):
        self.status_code = status
        self._cuerpo = cuerpo if cuerpo is not None else {
            "status": "success",
            "result": {"artist": "Oxia", "title": "Domino"}}
        self.text = str(self._cuerpo)

    def json(self):
        return self._cuerpo


@pytest.fixture
def audd_falso(monkeypatch):
    """Lo que se manda a la red, sin red."""
    import audd_helper
    mandado = []

    def _post(url, data=None, files=None, headers=None, timeout=None):
        f = files["file"]
        f = f[1] if isinstance(f, tuple) else f
        mandado.append({"url": url, "data": dict(data or {}),
                        "headers": dict(headers or {}), "bytes": f.read()})
        return _Resp()

    monkeypatch.setattr(audd_helper.requests, "post", _post)
    return mandado


def _token(client) -> str:
    r = client.post("/sync/register", json={
        "device_id": f"pc-{uuid.uuid4().hex[:12]}", "device_type": "windows"})
    assert r.status_code == 200, r.text
    return r.json()["device_token"]


def _trozo() -> bytes:
    return b"RIFF" + b"\x00" * 4000


def _pedir(client, token=None, via="analyze", cuerpo=None, ip="198.51.100.7",
           servicios="spotify,apple_music,cosas_raras"):
    cab = {"X-Forwarded-For": ip}
    if token:
        cab["X-Device-Token"] = token
    return client.post(
        "/audd/motor-local/identificar", headers=cab,
        data={"via": via, "return": servicios},
        files={"file": ("trozo.wav", io.BytesIO(cuerpo or _trozo()), "audio/wav")})


class TestRenderLaHacePorElMotor:
    def test_sin_token_del_aparato_no(self, client, audd_falso):
        assert _pedir(client).status_code == 401
        assert _pedir(client, token="inventado-0000").status_code == 401
        assert audd_falso == []

    def test_con_token_pasa_a_audd_con_el_de_render(self, client, app_mod,
                                                    audd_falso, monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "token-de-render")

        r = _pedir(client, token=_token(client), ip="198.51.100.8")

        assert r.status_code == 200, r.text
        assert r.json()["result"]["title"] == "Domino"
        [llamada] = audd_falso
        assert llamada["url"] == "https://api.audd.io/"
        assert llamada["data"]["api_token"] == "token-de-render"
        # Solo los servicios que AudD conoce.
        assert llamada["data"]["return"] == "spotify,apple_music"
        assert llamada["bytes"] == _trozo()

    def test_sin_token_de_audd_en_render_503(self, client, app_mod, audd_falso,
                                             monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "")
        assert _pedir(client, token=_token(client)).status_code == 503
        assert audd_falso == []

    def test_una_via_inventada_o_un_trozo_vacio_no(self, client, app_mod,
                                                   audd_falso, monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "token-de-render")
        token = _token(client)
        assert _pedir(client, token=token, via="otra").status_code == 400
        assert _pedir(client, token=token, cuerpo=b"x" * 10).status_code == 400
        assert audd_falso == []


class TestLosCupos:
    def test_la_cuenta_tiene_su_cupo_del_dia(self, client, app_mod, audd_falso,
                                             monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "token-de-render")
        monkeypatch.setattr(app_mod, "AUDD_MOTOR_LOCAL_CUENTA_DIA", 2)
        token = _token(client)

        codigos = [_pedir(client, token=token, ip="198.51.100.20").status_code
                   for _ in range(3)]

        assert codigos == [200, 200, 429]
        assert len(audd_falso) == 2, "lo que pasa del cupo no llega a AudD"
        # Otra cuenta sigue pudiendo.
        assert _pedir(client, token=_token(client),
                      ip="198.51.100.21").status_code == 200

    def test_la_ip_tiene_el_suyo(self, client, app_mod, audd_falso, monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "token-de-render")
        monkeypatch.setattr(app_mod, "AUDD_MOTOR_LOCAL_IP_DIA", 2)
        ip = "198.51.100.30"

        codigos = [_pedir(client, token=_token(client), ip=ip).status_code
                   for _ in range(3)]

        assert codigos == [200, 200, 429]

    def test_y_hay_un_tope_total(self, client, app_mod, audd_falso, monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "token-de-render")
        # Las del día, con la misma ventana que el cupo (otros tests apuntan
        # llamadas con relojes falsos).
        import time
        hechas = app_mod.db.cuenta_audd_por_render(time.time() - 86400)
        monkeypatch.setattr(app_mod, "AUDD_MOTOR_LOCAL_TOTAL_DIA", hechas + 1)

        assert _pedir(client, token=_token(client), ip="198.51.100.40").status_code == 200
        r = _pedir(client, token=_token(client), ip="198.51.100.41")
        assert r.status_code == 429 and r.json()["cupo"] == "total"

    def test_el_panel_las_cuenta_aparte(self, client, app_mod, audd_falso,
                                        monkeypatch):
        monkeypatch.setattr(app_mod, "AUDD_API_TOKEN", "token-de-render")
        antes = app_mod.db.resumen_audd_motor_local(30)["por_render"]
        assert _pedir(client, token=_token(client), ip="198.51.100.50").status_code == 200
        assert app_mod.db.resumen_audd_motor_local(30)["por_render"] == antes + 1


class TestElMotorLaPideARender:
    """`post_a_audd` en un motor local sin token propio."""

    @pytest.fixture
    def motor(self, monkeypatch, tmp_path):
        import audd_helper
        monkeypatch.setenv("LOCAL_ENGINE", "true")
        monkeypatch.setenv("RENDER_BACKEND_URL", "https://render.example")
        monkeypatch.setattr(audd_helper, "_token_del_aparato", "")
        trozo = tmp_path / "trozo.wav"
        trozo.write_bytes(_trozo())
        return audd_helper, str(trozo)

    def test_va_a_render_con_el_token_del_aparato(self, motor, audd_falso):
        audd_helper, trozo = motor
        audd_helper.apuntar_token_del_aparato("tok-del-aparato")

        r = audd_helper.post_a_audd(
            trozo, {"api_token": "", "return": "spotify,deezer,apple_music"},
            30, via="recognize")

        assert r.status_code == 200
        [llamada] = audd_falso
        assert llamada["url"] == "https://render.example/audd/motor-local/identificar"
        assert llamada["headers"] == {"X-Device-Token": "tok-del-aparato"}
        assert llamada["data"] == {"via": "recognize",
                                   "return": "spotify,deezer,apple_music"}
        assert "api_token" not in llamada["data"]

    def test_sin_token_del_aparato_no_sale_nada(self, motor, audd_falso):
        audd_helper, trozo = motor

        r = audd_helper.post_a_audd(trozo, {"api_token": ""}, 30, via="analyze")

        assert r.status_code == 401
        assert audd_falso == []

    def test_con_token_propio_va_directo(self, motor, audd_falso):
        """El que el DJ escribe en Ajustes, o una build anterior."""
        audd_helper, trozo = motor
        audd_helper.apuntar_token_del_aparato("tok-del-aparato")

        audd_helper.post_a_audd(trozo, {"api_token": "el-suyo", "return": ""},
                                30, via="identify")

        assert audd_falso[0]["url"] == "https://api.audd.io/"
        assert audd_falso[0]["data"]["api_token"] == "el-suyo"

    def test_el_motor_tiene_audd_sin_token_y_render_no(self, motor, monkeypatch):
        audd_helper, _ = motor
        assert audd_helper.hay_audd("") is True
        monkeypatch.setenv("LOCAL_ENGINE", "false")
        assert audd_helper.hay_audd("") is False
        assert audd_helper.hay_audd("x") is True

    def test_call_audd_tambien_pasa_por_render(self, motor, audd_falso, tmp_path):
        """La vía de /analyze (auto y «Limpiar metadata»)."""
        import numpy as np
        import soundfile as sf
        audd_helper, _ = motor
        audd_helper.apuntar_token_del_aparato("tok-del-aparato")
        tema = tmp_path / "tema.wav"
        sr = 22050
        t = np.arange(sr * 40) / sr
        sf.write(str(tema), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)

        track = audd_helper.call_audd(str(tema), "", timeout=5)

        assert track == {"artist": "Oxia", "title": "Domino"}
        assert audd_falso[0]["url"].endswith("/audd/motor-local/identificar")
        assert audd_falso[0]["data"]["via"] == "analyze"


class TestUnaSolaPuerta:
    def test_nadie_mas_llama_a_audd_a_pelo(self):
        aqui = os.path.dirname(os.path.abspath(__file__))
        fuera = []
        for nombre in os.listdir(aqui):
            if not nombre.endswith(".py") or nombre.startswith("test_"):
                continue
            with open(os.path.join(aqui, nombre), encoding="utf-8") as f:
                fuente = f.read()
            for m in re.finditer(r"requests\.post\(\s*['\"]https://api\.audd\.io", fuente):
                fuera.append(nombre)
        assert fuera == [], (
            f"{fuera} llaman a AudD sin `post_a_audd`: un motor local sin "
            "token se quedaría sin identificar")

    def test_el_motor_apunta_el_token_que_le_manda_la_app(self):
        import inspect
        import main
        fuente = inspect.getsource(main)
        bloque = fuente.split("async def apuntar_el_token_del_aparato")[1][:600]
        assert 'apuntar_token_del_aparato(request.headers.get("X-Device-Token"))' in bloque
        antes = fuente.split("async def apuntar_el_token_del_aparato")[0][-300:]
        assert "if IS_LOCAL_ENGINE:" in antes

    def test_apuntar_no_borra_uno_bueno_con_uno_vacio(self, monkeypatch):
        import audd_helper
        monkeypatch.setattr(audd_helper, "_token_del_aparato", "")
        audd_helper.apuntar_token_del_aparato("uno")
        audd_helper.apuntar_token_del_aparato("")
        audd_helper.apuntar_token_del_aparato(None)
        assert audd_helper.token_del_aparato() == "uno"


class TestElCupoLlenoNoEsNoLoConoce:
    def test_un_429_de_render_se_recuerda(self, monkeypatch, tmp_path):
        import audd_helper
        monkeypatch.setenv("LOCAL_ENGINE", "true")
        monkeypatch.setattr(audd_helper, "_token_del_aparato", "tok")
        monkeypatch.setattr(audd_helper, "_cupo_de_render_lleno_hasta", 0.0)
        trozo = tmp_path / "t.wav"
        trozo.write_bytes(_trozo())

        class _Lleno:
            status_code = 429
            headers = {"Retry-After": "3600"}
            text = '{"cupo": "cuenta"}'

        monkeypatch.setattr(audd_helper.requests, "post", lambda *a, **k: _Lleno())

        assert audd_helper.cupo_de_render_lleno() is False
        audd_helper.post_a_audd(str(trozo), {"api_token": ""}, 30, via="analyze")
        assert audd_helper.cupo_de_render_lleno() is True

    def test_y_limpiar_metadata_lo_dice_como_cupo(self):
        import inspect
        import main
        fuente = inspect.getsource(main)
        bloque = fuente.split("if force_audd and AUDD_AUTO_ENABLED and hay_audd(AUDD_API_TOKEN):")[1][:1500]
        assert "cupo_de_render_lleno()" in bloque
        assert "result.audd_daily_cap_reached = True" in bloque
