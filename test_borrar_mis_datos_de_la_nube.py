"""«Borrar mis datos de la nube» borra la CUENTA entera y no vuelve (2026-11-12).

La política de privacidad decía que «Borrar todo» de Ajustes borraba también
lo del servidor, y no era verdad: ese borrado es solo del aparato, y no había
ninguna forma de borrar del servidor la biblioteca de un usuario. Desde ese
día lo hace el usuario solo, desde Ajustes → Zona peligrosa, que llama a
`POST /sync/borrar-mis-datos`.

Lo que ata este test:

- se borra lo de la cuenta en sync.db y en analysis.db recorriendo el ESQUEMA
  DE VERDAD: una tabla nueva con `device_id`, `last_device_id`, `user_id` o
  `voter` que el borrado no conozca hace fallar `test_no_queda_nada...`;
- la otra cuenta no se toca;
- pide confirmar (400), un aparato registrado (403) y SU token siempre (401);
- después el sync contesta 410 a esos aparatos: registrarse otra vez no les
  abre una cuenta nueva por la que volver a subirlo todo; y repetir el borrado
  también da 410, que el cliente toma por hecho;
- lo que se calcula con lo borrado se rehace (los DJs de un tema, la media de
  estrellas, los votos de las notas de otros) y el ISRC que puso su identidad
  verificada se quita;
- la pasada del arranque recoge lo que un aparato borrado mandó después;
- el panel los cuenta;
- y `/sync/status`, que daba el id de los aparatos de todo el mundo a quien
  tuviera la firma de la app, pide el token de admin.

    pytest test_borrar_mis_datos_de_la_nube.py -v
"""
import os
import random
import sqlite3
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

# Las columnas que dicen de quién es una fila.
_DEL_APARATO = ("device_id", "last_device_id", "voter")
_DE_LA_CUENTA = ("user_id",)
# Lo ÚNICO que queda de una cuenta borrada, a propósito: la lápida.
_SE_QUEDAN = {"aparatos_borrados"}


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.pop("RENDER", None)
    import main
    import sync_endpoints

    sync_endpoints.SYNC_AUTH_SECRET = ""  # dev mode: sin firma
    return TestClient(main.app)


def _uid(tag):
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _registrar(client, device, tipo="macos-dmg"):
    r = client.post("/sync/register", json={"device_id": device,
                                            "device_type": tipo})
    assert r.status_code == 200, r.text
    return r.json()["user_id"], r.json()["device_token"]


def _cuenta_de_dos(client):
    """Un escritorio y un móvil vinculados: (user_id, pc, movil, tokens)."""
    pc, movil = _uid("pc"), _uid("movil")
    _, token_pc = _registrar(client, pc)
    _, token_movil = _registrar(client, movil, "ios")
    code = client.post("/sync/link/generate", json={"device_id": pc}).json()["code"]
    r = client.post("/sync/link/join", json={"device_id": movil, "code": code,
                                             "device_type": "mobile"})
    assert r.status_code == 200, r.text
    user_id = r.json()["user_id"]
    return user_id, pc, movil, {pc: token_pc, movil: token_movil}


def _borrar(client, device, token, confirmar=True):
    cabeceras = {"X-Device-Token": token} if token else {}
    return client.post("/sync/borrar-mis-datos",
                       json={"device_id": device, "confirmar": confirmar},
                       headers=cabeceras)


def _rutas():
    import main
    import sync_endpoints
    return sync_endpoints._DB_PATH, main.db.db_path


# ── El esquema de verdad ─────────────────────────────────────────────────────


def _tablas_con_dueno(conn):
    out = {}
    for (t,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"):
        if t.startswith("sqlite_") or t in _SE_QUEDAN:
            continue
        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
        if any(c[1] in _DEL_APARATO + _DE_LA_CUENTA for c in cols):
            out[t] = cols
    return out


def _meter_fila(conn, tabla, cols, aparato, cuenta):
    """Una fila de [aparato]/[cuenta] en [tabla], con lo que pida su esquema."""
    claves_pk = [c for c in cols if c[5]]
    valores = {}
    for _cid, nombre, tipo, notnull, defecto, pk in cols:
        t = (tipo or "").upper()
        if nombre in _DEL_APARATO:
            valores[nombre] = aparato
        elif nombre in _DE_LA_CUENTA:
            valores[nombre] = cuenta
        elif pk and t == "INTEGER" and len(claves_pk) == 1:
            continue  # rowid: lo pone SQLite
        elif notnull or pk:
            if defecto is not None and not pk:
                continue
            if "INT" in t:
                # En una clave, al azar (no chocar con las filas de verdad);
                # fuera, 1, que cabe en cualquier CHECK (estrellas de 1 a 5).
                valores[nombre] = random.randint(10 ** 6, 10 ** 9) if pk else 1
            elif "REAL" in t:
                valores[nombre] = 1.0
            else:
                valores[nombre] = f"x-{uuid.uuid4().hex[:10]}"
    nombres = ",".join(valores)
    marcas = ",".join("?" * len(valores))
    conn.execute(f"INSERT OR IGNORE INTO {tabla} ({nombres}) VALUES ({marcas})",
                 list(valores.values()))


def _filas_de(conn, tabla, cols, aparatos, cuenta):
    nombres = [c[1] for c in cols]
    donde, args = [], []
    for n in nombres:
        if n in _DEL_APARATO:
            donde.append(f"{n} IN ({','.join('?' * len(aparatos))})")
            args += aparatos
        elif n in _DE_LA_CUENTA:
            donde.append(f"{n} = ?")
            args.append(cuenta)
    return conn.execute(
        f"SELECT COUNT(*) FROM {tabla} WHERE {' OR '.join(donde)}",
        args).fetchone()[0]


def _sembrar(rutas, aparatos, cuenta):
    for ruta in rutas:
        conn = sqlite3.connect(ruta, timeout=30)
        try:
            for tabla, cols in _tablas_con_dueno(conn).items():
                for a in aparatos:
                    _meter_fila(conn, tabla, cols, a, cuenta)
            conn.commit()
        finally:
            conn.close()


def _contar(rutas, aparatos, cuenta):
    """{tabla: filas de [aparatos]/[cuenta]} en las dos BD."""
    out = {}
    for ruta in rutas:
        conn = sqlite3.connect(ruta, timeout=30)
        try:
            for tabla, cols in _tablas_con_dueno(conn).items():
                out[tabla] = _filas_de(conn, tabla, cols, aparatos, cuenta)
        finally:
            conn.close()
    return out


# ── Los tests ────────────────────────────────────────────────────────────────


def test_no_queda_nada_de_la_cuenta_y_la_otra_sigue(client):
    """Una fila de cada aparato en CADA tabla con dueño de las dos BD, y
    después del borrado ninguna es suya. La otra cuenta conserva las suyas."""
    rutas = _rutas()
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    otro = _uid("otro")
    otra_cuenta, _ = _registrar(client, otro)

    _sembrar(rutas, [pc, movil], cuenta)
    _sembrar(rutas, [otro], otra_cuenta)
    antes = _contar(rutas, [pc, movil], cuenta)
    vacias = [t for t, n in antes.items() if n == 0]
    assert not vacias, f"el test no supo meter filas en {vacias}"

    r = _borrar(client, movil, tokens[movil])
    assert r.status_code == 200, r.text
    assert r.json()["borrado"] is True
    assert r.json()["aparatos"] == 2, "la cuenta entera, no solo el móvil"

    quedan = {t: n for t, n in _contar(rutas, [pc, movil], cuenta).items() if n}
    assert not quedan, (
        f"quedan filas de la cuenta borrada en {quedan}: una tabla con dueño "
        "va en `_borrar_la_cuenta` (sync.db) o en `_TABLAS_DEL_BORRADO` "
        "(analysis.db)")

    del_otro = _contar(rutas, [otro], otra_cuenta)
    sin_nada = [t for t, n in del_otro.items() if n == 0]
    assert not sin_nada, f"el borrado se llevó lo de OTRA cuenta en {sin_nada}"


def test_el_gasto_de_identificar_se_queda_sin_el_aparato(client):
    import main
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    huella = _uid("fp")
    conn = sqlite3.connect(main.db.db_path)
    conn.execute("INSERT INTO audd_call_log (fingerprint, called_at, success, "
                 "source, device_id) VALUES (?, 1.0, 1, 'recognize', ?)",
                 (huella, pc))
    conn.commit()
    conn.close()

    assert _borrar(client, pc, tokens[pc]).status_code == 200

    conn = sqlite3.connect(main.db.db_path)
    filas = conn.execute("SELECT device_id FROM audd_call_log "
                         "WHERE fingerprint = ?", (huella,)).fetchall()
    conn.close()
    assert filas == [(None,)], "la llamada se queda (lo que costó), sin aparato"


def test_pide_confirmar_un_aparato_registrado_y_su_token(client):
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    otro = _uid("otro")
    _, token_otro = _registrar(client, otro)

    assert _borrar(client, pc, tokens[pc], confirmar=False).status_code == 400
    assert _borrar(client, _uid("nadie"), "x").status_code == 403
    assert _borrar(client, pc, None).status_code == 401, (
        "sin token, nunca: la firma de la app se saca del binario")
    assert _borrar(client, pc, token_otro).status_code == 401, (
        "el token de OTRO aparato no vale")
    assert _borrar(client, pc, tokens[movil]).status_code == 401, (
        "ni el de otro aparato de la misma cuenta: cada uno con el suyo")

    # Y nada de eso borró nada.
    r = client.get(f"/sync/pull/{pc}")
    assert r.status_code == 200, r.text


def test_despues_el_sync_contesta_410_y_no_se_vuelve_a_subir(client):
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    otro = _uid("otro")
    _registrar(client, otro)

    assert _borrar(client, pc, tokens[pc]).status_code == 200

    for aparato in (pc, movil):
        assert client.post("/sync/register", json={
            "device_id": aparato, "device_type": "macos-dmg"}).status_code == 410, (
            "registrarse otra vez le daría una cuenta nueva y lo volvería a "
            "subir todo")
        assert client.get(f"/sync/pull/{aparato}").status_code == 410
        assert client.get(f"/sync/pending/{aparato}").status_code == 410
        assert client.post("/sync/push", json={
            "device_id": aparato, "device_type": "ios", "data_type": "analysis",
            "items": []}).status_code == 410
        assert client.post("/sync/detected-track", json={
            "device_id": aparato, "artist": "a", "title": "b",
            "payload": {}, "detected_at": "2026-11-12T00:00:00Z",
        }).status_code == 410

    assert _borrar(client, pc, tokens[pc]).status_code == 410, (
        "repetirlo (el cliente no supo si llegó) da 410, que toma por hecho")

    # El de otra cuenta sigue como estaba.
    assert client.get(f"/sync/pull/{otro}").status_code == 200

    # Un aparato NUEVO (lo que hace la app tras borrar) empieza de cero.
    nuevo = _uid("nuevo")
    cuenta_nueva, _ = _registrar(client, nuevo)
    assert cuenta_nueva != cuenta


def test_lo_calculado_con_lo_borrado_se_rehace(client):
    """Los DJs de un tema, la media de estrellas y los votos de las notas de
    otros bajan con lo borrado; el ISRC que puso su identidad, se quita."""
    import main
    db = main.db
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    otro = _uid("otro")
    _registrar(client, otro)
    tema = _uid("tema")

    db.registrar_analista(tema, pc)
    db.registrar_analista(tema, otro)
    db.rate_track(tema, movil, 5)
    db.rate_track(tema, otro, 3)
    nota_del_otro = db.save_community_note(tema, otro, "drop a los 2:10")
    nota_suya = db.save_community_note(tema, pc, "cuidado con el break")
    db.upvote_community_note(nota_del_otro, movil)
    db.upvote_community_note(nota_suya, otro)

    # Su identidad verificada puso el ISRC en el tema.
    con_isrc = _uid("isrc")
    conn = sqlite3.connect(db.db_path)
    conn.execute("INSERT INTO tracks (id, fingerprint, artist, title, bpm, "
                 "analysis_json) VALUES (?, ?, 'A', 'B', 128, '{}')",
                 (con_isrc, con_isrc))
    conn.commit()
    conn.close()
    db.guardar_identidades(pc, [{"fingerprint": con_isrc, "artist": "A",
                                 "title": "B", "isrc": "ESA012345678"}])

    def popularidad():
        conn = sqlite3.connect(db.db_path)
        try:
            return conn.execute(
                "SELECT dj_count, avg_rating, total_ratings FROM "
                "track_popularity WHERE fingerprint = ?", (tema,)).fetchone()
        finally:
            conn.close()

    assert popularidad() == (2, 4.0, 2)

    assert _borrar(client, movil, tokens[movil]).status_code == 200

    assert popularidad() == (1, 3.0, 1), "queda solo lo del otro DJ"
    notas = {n["id"]: n for n in db.get_community_notes(tema)}
    assert nota_suya not in notas
    assert notas[nota_del_otro]["upvotes"] == 0, "su voto a la nota del otro"
    conn = sqlite3.connect(db.db_path)
    try:
        assert conn.execute("SELECT isrc FROM tracks WHERE id = ?",
                            (con_isrc,)).fetchone() == (None,)
        assert conn.execute("SELECT COUNT(*) FROM community_note_upvotes "
                            "WHERE note_id = ?", (nota_suya,)).fetchone()[0] == 0
    finally:
        conn.close()


def test_la_pasada_del_arranque_recoge_lo_que_llega_despues(client):
    """Un aparato de la cuenta con la app de antes no se entera del 410 y
    puede seguir mandando estrellas con su id viejo: lo quita la pasada que
    main.py corre al arrancar."""
    import main
    import sync_endpoints
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    assert _borrar(client, pc, tokens[pc]).status_code == 200
    assert movil in sync_endpoints.aparatos_borrados(dias=30)

    tema = _uid("tema")
    main.db.rate_track(tema, movil, 4)  # después del borrado
    main._repasar_aparatos_borrados()

    conn = sqlite3.connect(main.db.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM track_ratings WHERE "
                            "device_id = ?", (movil,)).fetchone()[0] == 0
        assert conn.execute("SELECT total_ratings FROM track_popularity WHERE "
                            "fingerprint = ?", (tema,)).fetchone() == (0,)
    finally:
        conn.close()


def test_el_panel_los_cuenta(client):
    import sync_endpoints
    from routes.admin_panel import _borrados

    antes = sync_endpoints.resumen_de_borrados(30)
    cuenta, pc, movil, tokens = _cuenta_de_dos(client)
    assert _borrar(client, pc, tokens[pc]).status_code == 200
    despues = _borrados()
    assert despues["cuentas_30d"] == antes["cuentas_30d"] + 1, (
        "una petición es una cuenta, con todos sus aparatos")
    assert despues["aparatos_30d"] == antes["aparatos_30d"] + 2
    assert despues["cuentas_total"] == antes["cuentas_total"] + 1


def test_status_ya_no_da_los_aparatos_de_todos(client, monkeypatch):
    import sync_endpoints
    monkeypatch.setattr(sync_endpoints, "ADMIN_TOKEN", "admin-de-prueba")
    assert client.get("/sync/status").status_code == 401
    r = client.get("/sync/status",
                   headers={"Authorization": "Bearer admin-de-prueba"})
    assert r.status_code == 200, r.text
    assert r.json()["total_items"] == sum(r.json()["by_type"].values())
