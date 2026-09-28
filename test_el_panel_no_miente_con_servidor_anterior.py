"""El panel da «Lo importado» y «AudD motores locales» de verdad.

En la lectura del embudo del 2026-09-27 las dos líneas decían «(servidor
anterior al 2026-09-26)» con el servidor al día: el Mac del owner acababa de
mandarle 4.911 votos. Las dos funciones del panel usaban un `db` que ese
módulo no tiene; saltaba un NameError, el `except` de red de seguridad se lo
tragaba y devolvían None, que `embudo.sh` lee como «el servidor no lo tiene».
Dos causas opuestas —servidor viejo y fallo nuestro— con el mismo texto.

Los tests de entonces miraban el FUENTE (`'lo_importado' in src`) o el método
de la base de datos, nunca lo que el panel devuelve. Estos piden el resumen al
panel.

    pytest test_el_panel_no_miente_con_servidor_anterior.py -v
"""

import ast
import os


def _panel():
    import main  # noqa: F401  (monta el singleton `main.db` que usa el panel)
    import routes.admin_panel as panel
    return panel


def test_lo_importado_sale_en_el_panel():
    resumen = _panel()._resumen_lo_importado()
    assert resumen is not None, 'None = embudo.sh dice «servidor anterior»'
    assert {'votos', 'huellas', 'aparatos', 'por_fuente'} <= set(resumen)


def test_el_audd_de_los_motores_locales_sale_en_el_panel():
    assert _panel()._resumen_audd_motor_local() is not None


def test_ninguna_funcion_del_panel_usa_un_db_que_no_existe():
    # El módulo no tiene `db` global: se pide con `_get_db()`. Una función que
    # lea `db` sin haberlo definido dentro revienta con NameError, y casi todas
    # las de aquí envuelven en un `except Exception` que lo convierte en un
    # dato vacío sin un solo error.
    import routes.admin_panel as panel
    with open(panel.__file__, encoding='utf-8') as fh:
        arbol = ast.parse(fh.read())
    globales = {n.id for n in ast.walk(arbol)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                and n.col_offset == 0}
    assert 'db' not in globales, 'si algún día hay un db global, revisa este test'
    rotas = []
    for fn in ast.walk(arbol):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        locales = {a.arg for a in fn.args.args}
        locales |= {n.id for n in ast.walk(fn)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if any(isinstance(n, ast.Name) and n.id == 'db'
               and isinstance(n.ctx, ast.Load) for n in ast.walk(fn)) \
                and 'db' not in locales:
            rotas.append(f'{fn.name} (línea {fn.lineno})')
    assert rotas == [], rotas


def test_el_script_distingue_servidor_viejo_de_fallo():
    # Para que la próxima vez no haga falta adivinar: sin la clave en la
    # respuesta es un servidor viejo; con la clave a None, el panel falló.
    ruta = os.path.join(os.path.dirname(__file__), 'routes', 'admin_panel.py')
    with open(ruta, encoding='utf-8') as fh:
        src = fh.read()
    assert '"lo_importado": _resumen_lo_importado()' in src
    assert '"motor_local_30d": _resumen_audd_motor_local()' in src
