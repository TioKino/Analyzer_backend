"""El panel admin no para a Render (2026-10-25).

Owner, tras la tanda D: «Usuarios, Errores y Técnico se quedan pensando mucho
tiempo hasta que sale un error». Render corre UN proceso de uvicorn, y esas
rutas eran `async def` que hacían su trabajo de SQLite en el event loop:
mientras /users o /stats recorrían `sync.db` entera, Render no atendía a nadie
más (tampoco a /analyze ni a Escuchar, ni a la pestaña Errores, que es ligera),
y la foto diaria del Resumen podía recorrer la biblioteca de todos a la vez.
`_tracks_por_device`, además, guardaba en memoria las filas de TODOS los
aparatos con su payload antes de contar.

Hoy: cada ruta corre en un hilo; las que recorren la biblioteca de todos (y la
foto del Resumen), además, de una en una; y el recuento por aparato va aparato
a aparato y se reusa mientras `sync_items` no cambie.

Y desde el 2026-10-28 las pesadas no esperan al recorrido: lo hace un
trabajador de fondo y la ruta contesta al momento con el último resultado (o
un 202 `calculando`). Con todo de uno en uno y recorridos de más de dos
minutos, cada pestaña esperaba a la anterior y el cliente se rendía a los
120 s.

    pytest test_el_panel_no_para_render.py -v
"""

import ast
import asyncio
import inspect
import json
import sqlite3
import threading
import time

import pytest
from starlette.requests import Request

import routes.admin_panel as ap

PESADAS = {
    "/users": "_desde_la_foto",
    "/stats": "_desde_la_foto",
    "/users/{device_id}/summary": "_desde_la_foto",
    "/telemetry": "_desde_la_foto",
    "/funnel": "_desde_la_foto",
    "/retention": "_desde_la_foto",
}
EN_UN_HILO = {
    "/activity": "_en_un_hilo",
    "/errors-grouped": "_en_un_hilo",
    "/errors": "_en_un_hilo",
    "/users/{device_id}/errors": "_en_un_hilo",
    "/catalogo/usuario/{device_id}": "_en_un_hilo",
}


def _rutas_get():
    arbol = ast.parse(inspect.getsource(ap))
    rutas = {}
    for n in arbol.body:
        if not isinstance(n, ast.AsyncFunctionDef):
            continue
        for d in n.decorator_list:
            if (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "get"
                    and d.args and isinstance(d.args[0], ast.Constant)):
                rutas[d.args[0].value] = n
    return rutas


@pytest.mark.parametrize("ruta,ayudante", sorted({**PESADAS, **EN_UN_HILO}.items()))
def test_la_ruta_solo_verifica_y_manda_el_trabajo_a_un_hilo(ruta, ayudante):
    nodo = _rutas_get()[ruta]
    cuerpo = [s for s in nodo.body
              if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(cuerpo) == 2, f"{ruta}: hace trabajo en el event loop"
    verifica, devuelve = cuerpo
    assert ast.unparse(verifica) == "await _verify_admin_secret(request)"
    assert isinstance(devuelve, ast.Return)
    assert isinstance(devuelve.value, ast.Await)
    assert devuelve.value.value.func.id == ayudante, ruta


@pytest.fixture
def sin_secreto(monkeypatch):
    monkeypatch.setattr(ap, "_get_admin_secret", lambda: "")
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)


def _peticion():
    return Request({"type": "http", "method": "GET", "headers": [],
                    "query_string": b""})


@pytest.fixture
def sin_fotos(monkeypatch):
    # conftest.py hace que la suite calcule al pedir; aquí se prueban las fotos.
    monkeypatch.setattr(ap, "CALCULAR_AL_PEDIR", False)
    monkeypatch.setattr(ap, "_fotos", {})
    monkeypatch.setattr(ap, "_fotos_pendientes", set())
    yield


def _esperar(cond, segundos=3.0):
    limite = time.monotonic() + segundos
    while time.monotonic() < limite:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _peticion_con(query=b""):
    return Request({"type": "http", "method": "GET", "headers": [],
                    "query_string": query})


def test_EL_CASO_usuarios_contesta_al_momento_aunque_tarde_minutos(
        sin_secreto, sin_fotos, monkeypatch):
    """Owner, 2026-10-28: «TimeoutException after 0:02:00» en Usuarios y en
    Técnico. La ruta ya no espera al recorrido: la primera vez dice que está
    calculando (202) y, en cuanto lo hay, da el resultado al momento."""
    monkeypatch.setattr(ap, "_list_users",
                        lambda request: time.sleep(0.3) or {"users": [1], "total": 1})
    t0 = time.monotonic()
    r = asyncio.run(ap.list_users(_peticion()))
    assert time.monotonic() - t0 < 0.2, "no espera al recorrido"
    assert r.status_code == 202 and json.loads(r.body)["calculando"] is True
    assert _esperar(lambda: ("users",) in ap._fotos)
    t0 = time.monotonic()
    r = asyncio.run(ap.list_users(_peticion()))
    assert time.monotonic() - t0 < 0.1
    assert r["users"] == [1] and r["_foto"]["tardo_s"] >= 0.3


def test_mientras_se_calcula_el_loop_sigue_atendiendo(sin_secreto, sin_fotos,
                                                      monkeypatch):
    monkeypatch.setattr(ap, "_telemetry", lambda request: time.sleep(0.6) or {})

    async def otra_peticion():
        t0 = time.monotonic()
        await asyncio.sleep(0.05)
        return time.monotonic() - t0

    async def las_dos():
        return await asyncio.gather(ap.telemetry(_peticion()), otra_peticion())

    _, espera = asyncio.run(las_dos())
    assert espera < 0.3, f"el loop estuvo parado {espera:.2f} s"


def test_una_foto_vieja_se_sirve_mientras_se_rehace(sin_secreto, sin_fotos,
                                                    monkeypatch):
    vuelta = {"n": 0}

    def calcular(request):
        vuelta["n"] += 1
        return {"n": vuelta["n"]}

    monkeypatch.setattr(ap, "_global_stats", calcular)
    ap._fotos[("stats",)] = {"valor": {"n": 0}, "cuando": time.time() - 10_000,
                             "tardo_s": 1.0, "error": None}
    r = asyncio.run(ap.global_stats(_peticion()))
    assert r["n"] == 0, "la vieja, al momento"
    assert _esperar(lambda: ap._fotos[("stats",)]["valor"]["n"] == 1)
    r = asyncio.run(ap.global_stats(_peticion()))
    assert r["n"] == 1 and r["_foto"]["hace_s"] < 5
    asyncio.run(ap.global_stats(_peticion()))
    assert vuelta["n"] == 1, "una nueva no se vuelve a calcular"


def test_cada_parametro_tiene_su_foto_y_no_se_pide_dos_veces(
        sin_secreto, sin_fotos, monkeypatch):
    llamadas = []
    cerrojo = threading.Event()

    def embudo(request, platform=None):
        llamadas.append(platform)
        cerrojo.wait(2)
        return {"platform": platform}

    monkeypatch.setattr(ap, "_funnel", embudo)
    for _ in range(3):
        asyncio.run(ap.funnel(_peticion(), platform="mobile"))
    asyncio.run(ap.funnel(_peticion(), platform="desktop"))
    cerrojo.set()
    assert _esperar(lambda: ("funnel", ("platform", "desktop")) in ap._fotos)
    assert sorted(llamadas) == ["desktop", "mobile"], "tres peticiones, un cálculo"
    r = asyncio.run(ap.funnel(_peticion(), platform="mobile"))
    assert r["platform"] == "mobile"


def test_esperar_es_para_los_scripts(sin_secreto, sin_fotos, monkeypatch):
    """`embudo.sh` necesita los datos en la primera petición."""
    monkeypatch.setattr(ap, "_retention", lambda request: time.sleep(0.2) or {"ok": 1})
    r = asyncio.run(ap.retention(_peticion_con(b"esperar=5")))
    assert r["ok"] == 1


def test_si_falla_se_queda_la_anterior_y_lo_dice(sin_secreto, sin_fotos,
                                                 monkeypatch):
    def roto(request):
        raise RuntimeError("sin disco")

    monkeypatch.setattr(ap, "_telemetry", roto)
    ap._fotos[("telemetry",)] = {"valor": {"ok": 1}, "cuando": time.time() - 10_000,
                                 "tardo_s": 1.0, "error": None}
    asyncio.run(ap.telemetry(_peticion()))
    assert _esperar(lambda: ap._fotos[("telemetry",)]["error"] is not None)
    r = asyncio.run(ap.telemetry(_peticion()))
    assert r["ok"] == 1 and "sin disco" in r["_foto"]["error"]


def test_las_pesadas_van_de_una_en_una():
    a_la_vez, maximo = [0], [0]
    cerrojo = threading.Lock()

    def recorrido():
        with cerrojo:
            a_la_vez[0] += 1
            maximo[0] = max(maximo[0], a_la_vez[0])
        time.sleep(0.05)
        with cerrojo:
            a_la_vez[0] -= 1

    async def tres():
        await asyncio.gather(*(ap._pesado(recorrido) for _ in range(3)))

    asyncio.run(tres())
    assert maximo[0] == 1


def test_el_trabajador_de_las_fotos_va_por_el_mismo_cerrojo(sin_fotos,
                                                            monkeypatch):
    visto = {}

    def calcular(request):
        visto["con_cerrojo"] = ap._cerrojo_pesado.locked()
        return {}

    ap._pedir_foto(("prueba",), calcular, {})
    assert _esperar(lambda: "con_cerrojo" in visto)
    assert visto == {"con_cerrojo": True}


def test_la_foto_del_resumen_va_por_el_mismo_cerrojo(monkeypatch):
    visto = {}

    def foto():
        visto["con_cerrojo"] = ap._cerrojo_pesado.locked()
        return {}

    monkeypatch.setattr(ap, "_hacer_foto_de_hoy", foto)
    monkeypatch.setattr(ap, "_resumen_en_marcha", False)
    assert ap._foto_en_segundo_plano()
    for _ in range(100):
        if "con_cerrojo" in visto:
            break
        time.sleep(0.01)
    assert visto == {"con_cerrojo": True}
    assert "_pesado(_hacer_foto_de_hoy)" in inspect.getsource(ap.resumen_recalcular)


# ── El recuento por aparato ───────────────────────────────────────────────────

@pytest.fixture
def sync_db(tmp_path, monkeypatch):
    ruta = str(tmp_path / "sync.db")
    conn = sqlite3.connect(ruta)
    conn.execute(
        "CREATE TABLE sync_items (key TEXT PRIMARY KEY, data_type TEXT NOT NULL, "
        "item_key TEXT NOT NULL, payload TEXT NOT NULL, deleted INTEGER DEFAULT 0, "
        "updated_at TEXT NOT NULL, last_device_id TEXT NOT NULL, "
        "device_type TEXT DEFAULT 'unknown', hash TEXT NOT NULL)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(ap, "_SYNC_DB_PATH", ruta)
    ap._por_device_guardado.update(clave=None, cuando=0.0, valor=None)
    yield ruta
    ap._por_device_guardado.update(clave=None, cuando=0.0, valor=None)


def _fila(ruta, aparato, item_key, payload, data_type="analysis"):
    conn = sqlite3.connect(ruta)
    conn.execute(
        "INSERT INTO sync_items (key, data_type, item_key, payload, updated_at, "
        "last_device_id, hash) VALUES (?, ?, ?, ?, '2026-10-25', ?, 'h')",
        (f"{aparato}|{data_type}|{item_key}", data_type, item_key,
         json.dumps(payload), aparato))
    conn.commit()
    conn.close()


def test_cuenta_por_aparato_con_los_dos_formatos(sync_db):
    for i in range(3):
        _fila(sync_db, "mac", f"t{i}", {"id": f"t{i}", "bpm": 128})
    _fila(sync_db, "pc", "all_analysis",
          {"tracks": {"a": {"id": "a"}, "b": {"id": "b"}}})
    _fila(sync_db, "movil", "s1", {"x": 1}, data_type="session")
    assert ap._tracks_por_device() == {"mac": 3, "pc": 2}


def test_aparato_a_aparato_sin_mirar_los_previews(sync_db, monkeypatch):
    _fila(sync_db, "mac", "t1", {"id": "t1"})
    _fila(sync_db, "pc", "t2", {"id": "t2"})
    llamadas = []
    original = ap._count_unique_tracks

    def espia(filas, check_previews=True):
        filas = list(filas)
        llamadas.append(({r["item_key"] for r in filas}, check_previews))
        return original(filas, check_previews=check_previews)

    monkeypatch.setattr(ap, "_count_unique_tracks", espia)
    ap._tracks_por_device()
    assert sorted(llamadas, key=lambda c: sorted(c[0])) == [
        ({"t1"}, False), ({"t2"}, False)], "una pasada por aparato, sin previews"


def test_se_reusa_mientras_no_cambia_y_se_rehace_si_cambia(sync_db, monkeypatch):
    _fila(sync_db, "mac", "t1", {"id": "t1"})
    pasadas = []
    original = ap._count_unique_tracks
    monkeypatch.setattr(
        ap, "_count_unique_tracks",
        lambda filas, check_previews=True: pasadas.append(1) or original(
            filas, check_previews=check_previews))
    assert ap._tracks_por_device() == {"mac": 1}
    assert ap._tracks_por_device() == {"mac": 1}
    assert len(pasadas) == 1, "la segunda vez no recorre nada"
    _fila(sync_db, "mac", "t2", {"id": "t2"})
    assert ap._tracks_por_device() == {"mac": 2}
    assert len(pasadas) == 2
    # Y caduca por tiempo (un blob legacy reescrito no cambia la clave).
    ap._por_device_guardado["cuando"] -= ap._POR_DEVICE_VALE_S + 1
    ap._tracks_por_device()
    assert len(pasadas) == 3


def test_quien_lo_recibe_no_puede_estropear_lo_guardado(sync_db):
    _fila(sync_db, "mac", "t1", {"id": "t1"})
    ap._tracks_por_device()["mac"] = 999
    assert ap._tracks_por_device() == {"mac": 1}
