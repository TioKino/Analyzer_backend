"""Los cortes finos de `investment` son los que permiten ELEGIR el limite del
plan gratuito.

Con solo `gte_100/500/1000` y una mediana de 4 tracks, entre la mediana y el
primer bucket hay un agujero donde vive casi todo el parque: cualquier limite
puesto ahi dentro se elige a ojo y no con la distribucion.
"""
from routes.admin_panel import _percentil


def test_percentil_de_lista_vacia_es_cero_y_no_revienta():
    assert _percentil([], 50) == 0
    assert _percentil([], 95) == 0


def test_percentil_devuelve_siempre_un_valor_QUE_EXISTE_en_la_lista():
    # Sin interpolar: un p90 de «312,5 tracks» describiria a un usuario que no
    # existe. El valor devuelto tiene que ser el de alguien.
    datos = sorted([1, 2, 3, 4, 900, 1000])
    for pct in (0, 25, 50, 75, 90, 95, 100):
        assert _percentil(datos, pct) in datos


def test_percentil_no_se_sale_por_ningun_extremo():
    datos = list(range(1, 11))  # 1..10, ya ordenado
    assert _percentil(datos, 0) == 1
    assert _percentil(datos, 100) == 10


def test_el_p50_coincide_con_la_mediana_en_una_lista_impar():
    datos = [1, 2, 3, 4, 5]
    assert _percentil(datos, 50) == 3


def test_el_p90_separa_la_cola_larga_de_la_mediana():
    # Forma real del parque: casi todos con pocos tracks y unos pocos con
    # bibliotecas enteras. La mediana no ve la cola; el p90 si.
    datos = sorted([2] * 90 + [1500] * 10)
    assert _percentil(datos, 50) == 2
    assert _percentil(datos, 95) == 1500


def test_los_buckets_son_acumulativos_y_nunca_crecen_al_subir_el_corte():
    counts = sorted([1, 5, 12, 30, 60, 120, 260, 700, 1200])
    b = {
        'gte_10': sum(1 for n in counts if n >= 10),
        'gte_25': sum(1 for n in counts if n >= 25),
        'gte_50': sum(1 for n in counts if n >= 50),
        'gte_100': sum(1 for n in counts if n >= 100),
        'gte_200': sum(1 for n in counts if n >= 200),
        'gte_500': sum(1 for n in counts if n >= 500),
        'gte_1000': sum(1 for n in counts if n >= 1000),
    }
    orden = ['gte_10', 'gte_25', 'gte_50', 'gte_100',
             'gte_200', 'gte_500', 'gte_1000']
    valores = [b[k] for k in orden]
    assert valores == sorted(valores, reverse=True), b
    # Y los cortes que ya existian no cambian de significado.
    assert b['gte_100'] == 4 and b['gte_500'] == 2 and b['gte_1000'] == 1


def test_el_contrato_de_investment_declara_los_siete_cortes():
    # Ata la forma de la respuesta: si alguien quita un corte, `embudo.sh` y la
    # decision del paywall se quedan sin el dato y no da ningun error.
    import io
    src = io.open('routes/admin_panel.py', encoding='utf-8').read()
    for corte in ('gte_10', 'gte_25', 'gte_50', 'gte_100',
                  'gte_200', 'gte_500', 'gte_1000'):
        assert src.count(f"'{corte}'") >= 2, corte
    for p in ('p50', 'p75', 'p90', 'p95'):
        assert f"'{p}'" in src, p
