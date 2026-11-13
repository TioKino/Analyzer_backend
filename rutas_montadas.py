"""Las rutas que la app tiene montadas DE VERDAD, valga la versión de FastAPI.

Hasta FastAPI 0.109 `app.include_router(...)` copiaba cada ruta del router a
`app.routes`, y los tests que comprueban qué está montado leían de ahí
(`{r.path for r in app.routes}`). Las versiones nuevas guardan el router
INCLUIDO como un solo objeto (con su `original_router` y el prefijo del
`include_router`), así que esa lectura dejaba fuera todo lo que no se declara
con `@app…` directamente: los tests fallaban con el servidor funcionando.

Esto lo recorre de las dos maneras. Solo lo usan los tests, pero vive fuera de
ellos porque lo comparten tres ficheros y `conftest.py` no se importa.
"""
from typing import Iterable, Set, Tuple

_METODOS_SIN_INTERES = ('HEAD', 'OPTIONS')


def rutas_montadas(app) -> Set[Tuple[str, str]]:
    """`{(método, ruta)}` de todo lo montado en [app], routers incluidos."""
    rutas: Set[Tuple[str, str]] = set()

    def andar(lista: Iterable, prefijo: str = '') -> None:
        for r in lista:
            path = getattr(r, 'path', None)
            metodos = getattr(r, 'methods', None)
            if path is not None and metodos:
                for m in metodos:
                    if m not in _METODOS_SIN_INTERES:
                        rutas.add((m, prefijo + path))
                continue
            original = getattr(r, 'original_router', None)
            if original is not None:
                contexto = getattr(r, 'include_context', None)
                andar(original.routes,
                      prefijo + (getattr(contexto, 'prefix', '') or ''))

    andar(app.routes)
    return rutas


def paths_montados(app) -> Set[str]:
    """Solo las rutas, sin el método."""
    return {path for _, path in rutas_montadas(app)}
