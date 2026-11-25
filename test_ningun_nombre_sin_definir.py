"""Ningún módulo del backend usa un nombre que no existe (2026-11-25).

Un nombre sin definir no da ningún error hasta que el programa pasa por esa
línea, y casi siempre está en un `except`: el sitio que solo corre cuando algo
ya ha fallado. Ahí el `NameError` sustituye al fallo de verdad, que no llega al
log, y si nadie más lo captura tumba lo que el `except` tenía que proteger.

Ha pasado dos veces:

  - 2026-09-27: dos funciones de `routes/admin_panel.py` usaban un `db` que ese
    módulo no tiene. El panel lo escondía con su `None` y `embudo.sh` lo leía
    como «servidor anterior» (`test_el_panel_no_miente_con_servidor_anterior.py`
    mira ese caso).
  - 2026-11-25: `database.py` usaba `logger` en el `except` del contador de
    AudD (`apuntar_audd_no_hizo_falta`) y no lo tenía. «Un contador no tumba
    un análisis», y no lo tumbaba solo porque `main.py` lo envuelve en otro
    `try`; el log decía `name 'logger' is not defined` en vez del fallo.

Se mira con `symtable`, de la librería estándar: cada nombre que el código lee
del módulo tiene que estar asignado o importado en el módulo, o ser de Python.
Sobre todo el backend da lo mismo que el F821 de pyflakes/ruff, sin añadir una
dependencia.

    pytest test_ningun_nombre_sin_definir.py -v
"""

import ast
import builtins
import logging
import os
import symtable

from database import AnalysisDB

AQUI = os.path.dirname(os.path.abspath(__file__))

# Lo que todo módulo tiene sin definirlo.
_DEL_MODULO = {'__name__', '__file__', '__doc__', '__spec__', '__loader__',
               '__package__', '__builtins__', '__path__', '__annotations__',
               '__dict__', '__cached__'}

# Carpetas que no son código del backend (además de las que empiezan por punto,
# como el `.venv` de la receta de CLAUDE.md).
_NO_SON_DEL_BACKEND = {'__pycache__', 'venv', 'env', 'node_modules',
                       'site-packages', 'build', 'dist'}


def nombres_sin_definir(src, nombre='<src>'):
    """Los nombres que el código lee del módulo y el módulo no tiene, cada uno
    con el sitio donde se lee. Un `from x import *` no se puede mirar: lo dice."""
    arbol = ast.parse(src, nombre)
    if any(isinstance(n, ast.ImportFrom) and any(a.name == '*' for a in n.names)
           for n in ast.walk(arbol)):
        return ['from … import * (no se puede mirar)']
    raiz = symtable.symtable(src, nombre, 'exec')
    definidos = set(dir(builtins)) | _DEL_MODULO
    leidos = []

    def recorrer(tabla):
        for s in tabla.get_symbols():
            n = s.get_name()
            if tabla is raiz:
                if s.is_assigned() or s.is_imported():
                    definidos.add(n)
                elif s.is_referenced():
                    leidos.append((n, 'el módulo'))
            else:
                # `global x` y asignarlo dentro de una función también lo define.
                if s.is_declared_global() and s.is_assigned():
                    definidos.add(n)
                if s.is_global() and s.is_referenced():
                    leidos.append((n, tabla.get_name()))
        for hija in tabla.get_children():
            recorrer(hija)

    recorrer(raiz)
    return sorted({f'{n} (en {donde})' for n, donde in leidos
                   if n not in definidos})


def _ficheros_del_backend():
    for carpeta, dirs, ficheros in os.walk(AQUI):
        dirs[:] = [d for d in dirs
                   if not d.startswith('.') and d not in _NO_SON_DEL_BACKEND]
        for f in ficheros:
            if f.endswith('.py'):
                yield os.path.join(carpeta, f)


def test_ningun_modulo_usa_un_nombre_sin_definir():
    malos = {}
    mirados = 0
    for ruta in _ficheros_del_backend():
        with open(ruta, encoding='utf-8') as fh:
            faltan = nombres_sin_definir(fh.read(), ruta)
        mirados += 1
        if faltan:
            malos[os.path.relpath(ruta, AQUI)] = faltan
    # Que el recorrido no se quede vacío por una carpeta mal excluida.
    assert mirados > 100, f'solo se han mirado {mirados} ficheros'
    assert not malos, 'nombres que el módulo no tiene:\n' + '\n'.join(
        f'  {f}: {", ".join(n)}' for f, n in sorted(malos.items()))


def test_el_detector_ve_lo_que_tiene_que_ver():
    """Si el detector dejara de ver algo, el test de arriba pasaría sin mirar
    nada. Lo que tiene que ver, y lo que no es un fallo."""
    def faltan(src):
        return [x.split(' ')[0] for x in nombres_sin_definir(src)]

    # El caso de database.py, y dónde más se esconde un nombre.
    assert faltan('def f():\n    try:\n        pass\n'
                  '    except Exception as e:\n        logger.warning(e)\n') == ['logger']
    assert faltan('def f():\n    def g():\n        return db.x\n    return g\n') == ['db']
    assert faltan('class A:\n    def m(self):\n        return Optional\n') == ['Optional']
    assert faltan('f = lambda: nada\n') == ['nada']
    assert faltan('def f(xs):\n    return [y for y in xs if y in faltan]\n') == ['faltan']
    assert faltan('if False:\n    print(cosa)\n') == ['cosa']
    # Nada de esto es un fallo.
    assert faltan('import logging\nlogger = logging.getLogger(__name__)\n'
                  'def f():\n    logger.warning(__file__)\n') == []
    assert faltan('def init():\n    global _cache\n    _cache = {}\n'
                  'def usa():\n    return _cache\n') == []
    assert faltan('try:\n    import numpy as np\nexcept ImportError:\n'
                  '    np = None\ndef f():\n    return np\n') == []
    assert faltan('def f():\n    return B()\nclass B:\n    pass\n') == []
    assert faltan('from os.path import *\n') == ['from']


def test_el_contador_de_audd_no_tumba_nada(tmp_path, caplog):
    """`apuntar_audd_no_hizo_falta` es best-effort: si no puede apuntar, lo
    dice en el log, con el fallo de verdad, y sigue."""
    db = AnalysisDB(db_path=str(tmp_path / 'analysis.db'))
    db.conn.execute('DROP TABLE audd_no_hizo_falta')
    db.conn.commit()
    with caplog.at_level(logging.WARNING, logger='database'):
        db.apuntar_audd_no_hizo_falta('identidad')   # no lanza
    assert 'contador de AudD fallo' in caplog.text
    assert 'no such table' in caplog.text, 'el log dice qué falló'
