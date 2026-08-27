"""La fase 2 del HMAC de escritura se decide con un numero, no a ojo.

`verify_write_auth` esta en fase 1: verifica la firma si viene, rechaza si
viene mal, y DEJA PASAR si no viene. La condicion escrita para pasar a fase 2
(`REQUIRE_WRITE_AUTH=1`) era «cuando los logs de Render muestren que casi nadie
llega sin firmar».

El problema: lo unico que existia era un `logger.info` por peticion. Para
decidir habia que rebuscar en los logs a mano, y los de Render rotan. O sea que
la decision se tomaba a ojo — y equivocarse devuelve **401 a todos los motores
locales ya publicados**, que no firman.

Es el mismo motivo por el que existe `sync_auth_stats` en el HMAC de sync, y de
ahi se copia el patron.

    pytest test_write_auth_medible.py -v
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


def _tabla(tmp_path, filas):
    conn = sqlite3.connect(str(tmp_path / 'w.db'))
    conn.execute(
        'CREATE TABLE write_auth_stats (day TEXT, signed INTEGER, '
        'path TEXT, n INTEGER, PRIMARY KEY (day, signed, path))'
    )
    conn.executemany('INSERT INTO write_auth_stats VALUES (?,?,?,?)', filas)
    conn.commit()
    return conn


def _resumen(conn):
    """La misma cuenta que hace `_write_auth_adoption`."""
    filas = conn.execute(
        'SELECT signed, path, SUM(n) FROM write_auth_stats GROUP BY signed, path'
    ).fetchall()
    firmadas = sum(int(r[2] or 0) for r in filas if r[0])
    sin_firmar = sum(int(r[2] or 0) for r in filas if not r[0])
    total = firmadas + sin_firmar
    return {
        'signed': firmadas,
        'unsigned': sin_firmar,
        'unsigned_pct': round(100.0 * sin_firmar / total, 1) if total else None,
    }


# ============================================================================
# EL NUMERO QUE DECIDE
# ============================================================================

def test_cuenta_firmadas_y_sin_firmar_por_separado(tmp_path):
    conn = _tabla(tmp_path, [
        ('2026-08-27', 1, '/cache-analysis', 80),
        ('2026-08-27', 0, '/cache-analysis', 20),
    ])
    try:
        r = _resumen(conn)
    finally:
        conn.close()

    assert r == {'signed': 80, 'unsigned': 20, 'unsigned_pct': 20.0}


def test_sin_trafico_el_pct_es_None_y_NO_cero(tmp_path):
    """0 % se leeria como «ya no llega nadie sin firmar» — justo la conclusion
    que activaria la fase 2 por error, dejando fuera a todo el parque.

    None se lee como «no hay datos», que es lo que de verdad pasa mientras la
    tabla esta vacia (se crea en la primera escritura tras el deploy).
    """
    conn = _tabla(tmp_path, [])
    try:
        assert _resumen(conn)['unsigned_pct'] is None
    finally:
        conn.close()


def test_el_desglose_por_endpoint_permite_cerrar_de_uno_en_uno(tmp_path):
    """Si lo que queda sin firmar es UN solo endpoint, se puede cerrar ese
    antes que el resto en vez de esperar a que todo llegue a cero."""
    conn = _tabla(tmp_path, [
        ('2026-08-27', 1, '/community/rate', 50),
        ('2026-08-27', 1, '/cache-analysis', 30),
        ('2026-08-27', 0, '/cache-analysis', 40),
    ])
    try:
        filas = conn.execute(
            'SELECT signed, path, SUM(n) FROM write_auth_stats '
            'GROUP BY signed, path'
        ).fetchall()
    finally:
        conn.close()

    por_path = {}
    for signed, path, n in filas:
        d = por_path.setdefault(path, {'signed': 0, 'unsigned': 0})
        d['signed' if signed else 'unsigned'] += int(n)

    assert por_path['/community/rate']['unsigned'] == 0, 'ese ya se puede exigir'
    assert por_path['/cache-analysis']['unsigned'] == 40


# ============================================================================
# EL CABLEADO
# ============================================================================

def test_se_cuentan_LOS_DOS_casos():
    """Contar solo las sin firmar daria un numero sin denominador: 40 sin
    firmar no dice nada si no sabes si el total son 45 o 45.000."""
    src = _src('main.py')
    assert '_record_write_auth(True, request.url.path)' in src
    assert '_record_write_auth(False, request.url.path)' in src


def test_contar_no_puede_tumbar_la_escritura():
    """Es una metrica de observacion: si la BD esta bloqueada o llena, la
    escritura colectiva tiene que seguir pasando."""
    src = _src('main.py')
    i = src.index('def _record_write_auth(')
    fn = src[i:i + 1800]
    assert 'try:' in fn
    assert 'except Exception:' in fn


def test_el_rechazo_por_firma_INVALIDA_no_se_cuenta_como_valido():
    """Una firma que viene y es incorrecta lanza 401 ANTES de contar: sumarla
    a `signed` inflaria la adopcion con intentos fallidos y adelantaria la
    fase 2."""
    src = _src('main.py')
    i = src.index('raise HTTPException(401, "Invalid signature")')
    j = src.index('_record_write_auth(True', i)
    entre = src[i:j]
    assert 'return True' not in entre.split('\n')[0]
    # El registro va DESPUES del raise, no antes.
    assert j > i


def test_el_panel_lo_expone():
    src = _src('routes/admin_panel.py')
    assert '"write_auth_30d": _write_auth_adoption(days=30),' in src
    assert 'def _write_auth_adoption(' in src


def test_la_tabla_ausente_no_rompe_el_panel():
    """Se crea en la primera escritura colectiva tras el deploy. Hasta
    entonces el panel entero no puede caerse por eso — ya paso con un OOM en
    admin."""
    src = _src('routes/admin_panel.py')
    i = src.index('def _write_auth_adoption(')
    fn = src[i:i + 2500]
    assert 'except sqlite3.OperationalError:' in fn
    assert 'return {}' in fn
