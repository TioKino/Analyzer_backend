"""«Dar todo por visto» en la pestaña Errores del panel (2026-10-28).

Owner: «la pantalla errores parece funcionar, ¿la limpio? Aparecen errores
antiguos». Había que resolverlos grupo a grupo. Hoy un botón los marca todos
como resueltos hasta ese momento: no se borra nada, y un error que vuelve a
pasar entra como una fila nueva sin resolver, así que su grupo sale otra vez.

    pytest test_dar_los_errores_por_vistos.py -v
"""

import inspect

from database import AnalysisDB


def _db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / "errores.db"))


def _error(db, msg="boom", cuando=None):
    i = db.log_analysis_error(device_id="d1", filename="a.mp3", fingerprint=None,
                              error_class="ValueError", error_msg=msg)
    if cuando:
        conn = db._open_conn()
        conn.execute("UPDATE analysis_errors SET timestamp = ? WHERE id = ?",
                     (cuando, i))
        conn.commit()
        conn.close()
    return i


def test_resuelve_lo_de_antes_y_no_borra_nada(tmp_path):
    db = _db(tmp_path)
    _error(db, "viejo", "2026-09-01 10:00:00")
    _error(db, "viejo", "2026-10-20 10:00:00")
    _error(db, "de ayer", "2026-10-27 12:00:00")
    assert db.count_errors(resolved=False) == 3
    assert db.resolve_errors_before("2026-10-27T00:00:00Z") == 2
    assert db.count_errors(resolved=False) == 1
    assert db.count_errors(resolved=True) == 2, "resolver no es borrar"
    # Sin fecha: todo lo que hay hasta ahora.
    assert db.resolve_errors_before() == 1
    assert db.count_errors(resolved=False) == 0


def test_un_error_que_vuelve_sale_otra_vez(tmp_path):
    db = _db(tmp_path)
    _error(db, "se repite", "2026-10-01 10:00:00")
    db.resolve_errors_before()
    assert db.get_errors_grouped(resolved=False) == []
    _error(db, "se repite")
    grupos = db.get_errors_grouped(resolved=False)
    assert len(grupos) == 1


def test_la_ruta_corre_en_un_hilo():
    import routes.admin_panel as ap
    src = inspect.getsource(ap.resolve_all_errors)
    assert "_verify_admin_secret" in src
    assert "_en_un_hilo(" in src
