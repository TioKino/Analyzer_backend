"""Las portadas se guardan a un tamaño razonable (2026-11-19).

49 GB de portadas a resolución completa llenaban el disco de Render
(`portadas.py`). Este test comprueba que una grande sale a como mucho
`LADO_MAX` px en JPEG, que una pequeña, un GIF o algo ilegible no se tocan,
que la pasada sobre la carpeta cambia el PNG por un JPG sin dejar el viejo y
que medir no escribe nada.

    pytest test_las_portadas_se_reducen.py -v
"""
import io
import os
import random

from PIL import Image

import portadas


def _imagen(lado, formato='PNG', alfa=False, ruido=True):
    modo = 'RGBA' if alfa else 'RGB'
    img = Image.new(modo, (lado, lado), (40, 80, 120, 128) if alfa else (40, 80, 120))
    if ruido:
        rnd = random.Random(1)
        px = img.load()
        for _ in range(lado * lado // 3):
            x, y = rnd.randrange(lado), rnd.randrange(lado)
            px[x, y] = tuple(rnd.randrange(256) for _ in modo)
    b = io.BytesIO()
    img.save(b, formato, **({'quality': 98} if formato == 'JPEG' else {}))
    return b.getvalue()


def test_una_grande_sale_pequena_y_en_jpeg():
    grande = _imagen(1400)
    nuevo = portadas.reducir(grande)
    assert nuevo is not None and len(nuevo) < len(grande)
    with Image.open(io.BytesIO(nuevo)) as img:
        assert img.format == 'JPEG'
        assert max(img.size) == portadas.LADO_MAX


def test_la_transparencia_se_aplana_y_no_revienta():
    nuevo = portadas.reducir(_imagen(1200, alfa=True))
    assert nuevo is not None
    with Image.open(io.BytesIO(nuevo)) as img:
        assert img.mode == 'RGB'


def test_lo_que_no_hay_que_tocar_no_se_toca():
    assert portadas.reducir(_imagen(200, 'JPEG', ruido=False)) is None, 'pequeña'
    assert portadas.reducir(b'GIF89a' + b'\x00' * 100_000) is None, 'un gif'
    assert portadas.reducir(b'\xff\xd8\xff' + os.urandom(100_000)) is None, 'rota'


def test_la_pasada_cambia_el_png_por_un_jpg_y_medir_no_escribe(tmp_path):
    (tmp_path / 'aaa.png').write_bytes(_imagen(1400))
    (tmp_path / 'bbb.jpg').write_bytes(_imagen(200, 'JPEG', ruido=False))
    (tmp_path / 'ccc.gif').write_bytes(b'GIF89a' + b'\x00' * 100_000)
    antes = sorted(os.listdir(tmp_path))

    m = portadas.medir(str(tmp_path), muestra=10, semilla=1)
    assert sorted(os.listdir(tmp_path)) == antes, 'medir no escribe nada'
    assert m['ficheros'] == 3 and m['muestra_cambiarian'] == 1
    assert m['ahorro_estimado'] > 0

    r = portadas.Reduccion(str(tmp_path), pausa_s=0)
    r.correr()
    assert sorted(os.listdir(tmp_path)) == ['aaa.jpg', 'bbb.jpg', 'ccc.gif']
    assert r.estado['reducidos'] == 1 and r.estado['errores'] == 0
    with Image.open(tmp_path / 'aaa.jpg') as img:
        assert max(img.size) == portadas.LADO_MAX

    r.correr()
    assert r.estado['reducidos'] == 0, 'lo ya reducido no se vuelve a tocar'


def test_las_dos_puertas_guardan_reducido(tmp_path, monkeypatch):
    import artwork_and_cuepoints as aac
    monkeypatch.setattr(aac, 'ARTWORK_CACHE_DIR', str(tmp_path))
    (tmp_path / 'fff.png').write_bytes(b'vieja')
    nombre = aac.save_artwork_to_cache('fff', _imagen(1400), 'image/png')
    assert nombre == 'fff.jpg'
    assert os.listdir(tmp_path) == ['fff.jpg'], 'sin la vieja de otra extensión'
    src = open('routes/analysis_artwork.py', encoding='utf-8').read()
    assert 'portadas.guardar' in src, 'la subida de una portada también reduce'


# ---- Upload2Shell (CVE-2026-125396, aviso de Render del 2026-11-21) --------
# Pillow pasa un EPS/PostScript a Ghostscript. Una portada puede salir de las
# etiquetas de cualquier audio subido a /analyze, así que `reducir` solo abre
# los cuatro formatos de una portada y Ghostscript no se llama nunca.

def _eps(bytes_minimos):
    cabecera = (b'%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 2000 2000\n'
                b'%%EndComments\n')
    # Líneas cortas: Pillow no lee como EPS una línea de más de 255 bytes.
    return cabecera + b'% relleno\n' * (bytes_minimos // 10 + 1)


def test_un_eps_no_llega_a_ghostscript(monkeypatch):
    from PIL import EpsImagePlugin
    llamadas = []
    monkeypatch.setattr(EpsImagePlugin, 'Ghostscript',
                        lambda *a, **k: llamadas.append(a) or None)
    datos = _eps(portadas.BYTES_MIN)
    assert portadas.reducir(datos) is None
    assert llamadas == [], 'Pillow le ha pasado el EPS a Ghostscript'


def test_pillow_sin_la_restriccion_si_lo_pasaria(monkeypatch):
    """Que el test de arriba no pase por casualidad: el mismo EPS, abierto
    como lo hacía `reducir` antes, sí llega a Ghostscript."""
    from PIL import EpsImagePlugin
    llamadas = []

    def _gs(*a, **k):
        llamadas.append(a)
        raise OSError('Ghostscript de mentira')
    monkeypatch.setattr(EpsImagePlugin, 'Ghostscript', _gs)
    with Image.open(io.BytesIO(_eps(portadas.BYTES_MIN))) as img:
        try:
            img.load()
        except OSError:
            pass
    assert llamadas, 'este Pillow ya no abre EPS: revisa el test de arriba'


def test_las_portadas_de_verdad_se_siguen_abriendo():
    for formato in ('JPEG', 'PNG', 'WEBP'):
        datos = _imagen(1500, formato)
        assert portadas.reducir(datos) is not None, formato
