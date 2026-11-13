"""El mantenimiento diario de Render (auditoría S8, 2026-11-13): copia de las
dos bases de datos, cuentas abandonadas fuera y errores viejos podados.
Ver `mantenimiento.py`."""
import gzip
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import mantenimiento as m


def _base(ruta, filas=50):
    conn = sqlite3.connect(ruta)
    conn.execute("CREATE TABLE t (x TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [("fila %d" % i,) for i in range(filas)])
    conn.commit()
    conn.close()


def test_la_copia_se_puede_abrir_y_solo_se_guardan_dos(tmp_path):
    a, s = tmp_path / "analysis.db", tmp_path / "sync.db"
    _base(a)
    _base(s, 7)
    copias = tmp_path / "copias"
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    for i in range(4):
        m.hacer_copia({"analysis": str(a), "sync": str(s)}, str(copias),
                      ahora=t0 + timedelta(days=7 * i))
    hechas = m.copias_hechas(str(copias))
    assert hechas == ["2026-11-15", "2026-11-22"]
    # La copia es una base de datos de verdad, entera.
    gz = copias / "2026-11-22" / "sync.db.gz"
    plano = tmp_path / "vuelta.db"
    plano.write_bytes(gzip.open(gz).read())
    assert sqlite3.connect(plano).execute("SELECT COUNT(*) FROM t").fetchone()[0] == 7
    assert not any(n.endswith(".tmp") for n in os.listdir(copias))


def test_toca_cada_semana(tmp_path):
    copias = tmp_path / "copias"
    assert m.toca_copia(str(copias)) is True
    (copias / "2026-11-10").mkdir(parents=True)
    hoy = datetime(2026, 11, 13, tzinfo=timezone.utc)
    assert m.toca_copia(str(copias), hoy) is False
    assert m.toca_copia(str(copias), hoy + timedelta(days=4)) is True


def test_sin_sitio_no_se_hace_ni_deja_restos(tmp_path, monkeypatch):
    a = tmp_path / "analysis.db"
    _base(a)
    copias = tmp_path / "copias"

    class Lleno:
        free = 0

    monkeypatch.setattr(m.shutil, "disk_usage", lambda _p: Lleno)
    with pytest.raises(RuntimeError):
        m.hacer_copia({"analysis": str(a)}, str(copias))
    assert m.copias_hechas(str(copias)) == []
    assert os.listdir(copias) == []


def test_caduca_solo_lo_que_lleva_24_meses_sin_verse():
    llamadas = []
    cortes = []

    def sin_ver(corte, limite):
        cortes.append(corte)
        return [{"user_id": "u1", "aparatos": ["d1", "d2"]}]

    n = m.caducar_cuentas(sin_ver, lambda u, a: llamadas.append((u, a)) or {},
                          ahora=datetime(2028, 12, 1, tzinfo=timezone.utc))
    assert n == 1 and llamadas == [("u1", ["d1", "d2"])]
    assert cortes[0].startswith("2026-12")


class _Db:
    def __init__(self):
        self.args = None

    def prune_old_errors(self, days, only_resolved):
        self.args = (days, only_resolved)
        return 3


def test_una_parte_que_falla_no_para_las_demas(tmp_path):
    db = _Db()

    def roto(*_):
        raise RuntimeError("sync.db no contesta")

    r = m.pasada({"analysis": str(tmp_path / "no-existe.db")},
                 str(tmp_path / "copias"), db, roto, lambda *_: {})
    assert "caducidad_error" in r
    assert r["errores"] == 3
    # Los errores sin resolver también se podan.
    assert db.args == (m.DIAS_DE_UN_ERROR, False)


def test_las_cuentas_sin_ver_se_buscan_en_sync_db(tmp_path, monkeypatch):
    import sync_endpoints as se
    ruta = str(tmp_path / "sync.db")
    conn = sqlite3.connect(ruta)
    conn.execute("CREATE TABLE user_devices (device_id TEXT PRIMARY KEY, "
                  "user_id TEXT, linked_at TEXT, visto_at TEXT)")
    viejo = "2024-01-01T00:00:00"
    nuevo = "2026-11-01T00:00:00"
    conn.executemany("INSERT INTO user_devices VALUES (?,?,?,?)", [
        ("a1", "vieja", viejo, viejo), ("a2", "vieja", viejo, None),
        ("b1", "mixta", viejo, viejo), ("b2", "mixta", viejo, nuevo),
    ])
    conn.commit()
    conn.close()
    monkeypatch.setattr(se, "_DB_PATH", ruta)
    r = se.cuentas_sin_ver_desde("2025-01-01T00:00:00")
    # Una cuenta con UN aparato reciente no caduca.
    assert [c["user_id"] for c in r] == ["vieja"]
    assert sorted(r[0]["aparatos"]) == ["a1", "a2"]


def test_main_lo_arranca_solo_en_render():
    src = open(os.path.join(os.path.dirname(__file__), "main.py")).read()
    assert 'if os.environ.get("RENDER") and not IS_LOCAL_ENGINE:' in src
    assert "_mantenimiento.arrancar(" in src
