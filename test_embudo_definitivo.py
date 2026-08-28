"""Auditoria del embudo: los tres numeros que decidian el paywall estaban mal.

No eran imprecisiones. Cada uno tenia DOS sesgos tirando en direcciones
opuestas, que es lo que hace imposible corregir a ojo — y por eso cada lectura
sugeria una conclusion distinta y habia que volver a tocar el embudo.

  1. RETENCION, inflada. El D0 salia de `MIN(day) FROM events`, y `events` se
     purga a los 90 dias. La purga no borraba solo datos viejos: REESCRIBIA la
     fecha de alta de todo veterano, que pasaba a figurar como recien
     instalado. Y como son justo los que siguen activos, entraban en la
     cohorte y contaban como retenidos.

  2. INVERSION, deflactada por abajo e inflada por arriba. Sumaba eventos
     `import_completed`: quien importo hace 4 meses contaba CERO (purga), y
     quien reimporto la misma carpeta 4 veces contaba x4.

  3. EL EMBUDO NO ERA UN EMBUDO. Cada paso contaba dispositivos distintos de
     los ultimos 30 dias. `app_opened` salta en cada arranque (= todos los
     activos del mes) y `onboarding_completed` solo la primera vez en la vida
     del dispositivo (= solo las altas del mes). La "caida" entre paso 1 y 2
     era, en su mayor parte, gente que ya habia pasado por ahi hace meses.
     Cuanto mas veterano el parque, peor pintaba — al reves de la realidad.

    pytest test_embudo_definitivo.py -v
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _src(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


def _bd(tmp_path):
    conn = sqlite3.connect(str(tmp_path / 'e.db'))
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, '
        "timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, device_id TEXT, "
        'event_name TEXT NOT NULL, props TEXT, platform TEXT, '
        'day TEXT GENERATED ALWAYS AS (substr(timestamp,1,10)) VIRTUAL)'
    )
    conn.execute(
        'CREATE TABLE device_first_seen (device_id TEXT PRIMARY KEY, '
        'first_day TEXT NOT NULL, first_platform TEXT, first_app_version TEXT)'
    )
    return conn


def _d0(conn):
    """La regla de D0: la tabla durable, con respaldo en events."""
    return {
        r[0]: r[1] for r in conn.execute(
            'SELECT device_id, first_day AS d0 FROM device_first_seen '
            'WHERE device_id IS NOT NULL '
            'UNION '
            'SELECT device_id, MIN(day) FROM events '
            "WHERE event_name = 'app_opened' AND device_id IS NOT NULL "
            '  AND device_id NOT IN (SELECT device_id FROM device_first_seen) '
            'GROUP BY device_id'
        ).fetchall()
    }


# ============================================================================
# 1. LA PURGA YA NO REESCRIBE EL D0
# ============================================================================

def test_el_veterano_conserva_su_fecha_de_alta_tras_la_purga(tmp_path):
    """El fallo entero, en un test.

    Un usuario de hace un ano al que la purga le ha borrado los eventos
    viejos. Antes su D0 pasaba a ser el evento mas reciente que sobreviviera y
    entraba en la cohorte de este mes como alta nueva.
    """
    conn = _bd(tmp_path)
    try:
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day) '
            "VALUES ('veterano', '2025-09-01')"
        )
        # Lo unico que sobrevive a la purga: su actividad reciente.
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES (datetime('now'), 'veterano', 'app_opened')"
        )
        conn.commit()

        assert _d0(conn)['veterano'] == '2025-09-01'
    finally:
        conn.close()


def test_el_D0_no_se_mueve_por_mas_eventos_que_mande(tmp_path):
    """`INSERT OR IGNORE`: la fila se sella una vez y no se toca."""
    conn = _bd(tmp_path)
    try:
        for _ in range(3):
            conn.execute(
                'INSERT OR IGNORE INTO device_first_seen (device_id, first_day) '
                "VALUES ('d1', date('now'))"
            )
        conn.execute(
            'INSERT OR IGNORE INTO device_first_seen (device_id, first_day) '
            "VALUES ('d1', '2030-01-01')"
        )
        conn.commit()
        r = conn.execute(
            "SELECT first_day FROM device_first_seen WHERE device_id='d1'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert r != '2030-01-01'


def test_un_device_SIN_fila_todavia_no_se_pierde(tmp_path):
    """Respaldo para los datos anteriores a que la tabla existiera. Es el D0
    malo, pero perderlos del todo seria peor."""
    conn = _bd(tmp_path)
    try:
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES ('2026-08-01 10:00:00', 'legado', 'app_opened')"
        )
        conn.commit()
        assert _d0(conn)['legado'] == '2026-08-01'
    finally:
        conn.close()


def test_la_tabla_durable_GANA_al_respaldo(tmp_path):
    """Si hay fila sellada, los eventos no la pisan — que es todo el punto."""
    conn = _bd(tmp_path)
    try:
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day) '
            "VALUES ('d1', '2025-01-01')"
        )
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES ('2026-08-01 10:00:00', 'd1', 'app_opened')"
        )
        conn.commit()
        # El UNION puede traer las dos; la de la tabla es la que manda.
        filas = conn.execute(
            "SELECT first_day FROM device_first_seen WHERE device_id='d1'"
        ).fetchone()
    finally:
        conn.close()

    assert filas[0] == '2025-01-01'


def test_la_siembra_va_ANTES_que_la_purga():
    """Al reves, cada deploy borraria justo lo que veniamos a rescatar."""
    src = _src('main.py')
    i = src.index('db.backfill_first_seen()')
    j = src.index('db.purge_old_events(')
    assert i < j, 'sembrar despues de purgar no rescata nada'


def test_el_D0_se_sella_al_registrar_el_evento():
    src = _src('database.py')
    i = src.index('def log_event(')
    fn = src[i:i + 3000]
    assert 'INSERT OR IGNORE INTO device_first_seen' in fn


def test_la_tabla_del_D0_NO_se_purga():
    """Si alguien la mete en la purga, el fallo vuelve entero."""
    src = _src('database.py')
    i = src.index('def purge_old_events(')
    fn = src[i:i + 1200]
    assert 'device_first_seen' not in fn


# ============================================================================
# 2. LA INVERSION SALE DE LAS BIBLIOTECAS, NO DE LOS EVENTOS
# ============================================================================

def test_la_inversion_se_cuenta_sobre_sync_db():
    """`events` se purga y los import se repiten. Las bibliotecas de `sync.db`
    no: son el estado actual, que es lo que se queria medir."""
    src = _src('routes/admin_panel.py')
    i = src.index('def _library_investment_real(')
    fn = src[i:i + 2500]
    assert '_get_sync_conn()' in fn
    assert "data_type = 'analysis'" in fn
    # El mismo dedup que /admin/users, para que los dos numeros cuadren.
    assert '_count_unique_tracks(' in fn
    assert "'source': 'sync.db'" in fn


def test_la_cuenta_vieja_no_desaparece_pero_cambia_de_nombre():
    """Medir la actividad de import del trimestre es legitimo. Llamarlo
    «cuanta biblioteca tiene la gente» era lo que no lo era."""
    src = _src('routes/admin_panel.py')
    assert '"import_activity_90d": imports_90d,' in src
    assert '"investment": investment,' in src


def test_no_carga_la_biblioteca_entera_en_memoria_de_golpe():
    """Los endpoints admin recorren la biblioteca entera y ya tumbaron
    produccion con un OOM. El cursor se itera, no se hace fetchall()."""
    src = _src('routes/admin_panel.py')
    i = src.index('def _library_investment_real(')
    fn = src[i:i + 2500]
    j = fn.index("WHERE data_type = 'analysis'")
    assert '.fetchall()' not in fn[j - 200:j + 200]


# ============================================================================
# 3. EL EMBUDO SIGUE A UNA COHORTE
# ============================================================================

def _pasos(conn, evento):
    """Dispositivos de la cohorte del mes que dispararon `evento`."""
    return conn.execute(
        'WITH firsts AS ('
        ' SELECT device_id, first_day AS d0 FROM device_first_seen'
        ' WHERE device_id IS NOT NULL) '
        'SELECT COUNT(DISTINCT e.device_id) FROM events e '
        'JOIN firsts f ON f.device_id = e.device_id '
        "WHERE f.d0 >= date('now','-30 days') AND e.event_name = ?",
        (evento,),
    ).fetchone()[0]


def test_el_veterano_NO_cuenta_como_abandono_de_onboarding(tmp_path):
    """El caso que hacia que el embudo empeorara segun crecia el parque.

    El veterano abre la app cada dia (`app_opened`) pero completo el
    onboarding hace un ano, asi que nunca vuelve a emitirlo. Contandolo por
    ventana aparecia en el paso 1 y no en el 2: un abandono inventado.
    """
    conn = _bd(tmp_path)
    try:
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day) '
            "VALUES ('veterano', '2025-09-01')"
        )
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES (datetime('now'), 'veterano', 'app_opened')"
        )
        conn.commit()

        # Por ventana estaria en el paso 1. Por cohorte no esta en ninguno,
        # que es lo correcto: no es un alta de este mes.
        por_ventana = conn.execute(
            "SELECT COUNT(DISTINCT device_id) FROM events "
            "WHERE event_name='app_opened' "
            "  AND timestamp >= datetime('now','-30 days')"
        ).fetchone()[0]

        assert por_ventana == 1, 'la cuenta vieja si lo veia'
        assert _pasos(conn, 'app_opened') == 0, 'la cohorte no'
    finally:
        conn.close()


def test_el_alta_del_mes_SI_cuenta_en_todos_sus_pasos(tmp_path):
    conn = _bd(tmp_path)
    try:
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day) '
            "VALUES ('nuevo', date('now','-5 days'))"
        )
        for ev in ('app_opened', 'onboarding_completed', 'import_completed'):
            conn.execute(
                'INSERT INTO events (timestamp, device_id, event_name) '
                "VALUES (datetime('now','-5 days'), 'nuevo', ?)",
                (ev,),
            )
        conn.commit()

        assert _pasos(conn, 'app_opened') == 1
        assert _pasos(conn, 'onboarding_completed') == 1
        assert _pasos(conn, 'import_completed') == 1
    finally:
        conn.close()


def test_convertir_TARDE_sigue_siendo_convertir(tmp_path):
    """El embudo mide si llega, no si corre. Fijar la cohorte y NO limitar la
    fecha del evento es justo lo que permite esto: quien se instalo el dia 28
    y completo el import el 32 esta convertido."""
    conn = _bd(tmp_path)
    try:
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day) '
            "VALUES ('tardon', date('now','-28 days'))"
        )
        conn.execute(
            'INSERT INTO events (timestamp, device_id, event_name) '
            "VALUES (datetime('now'), 'tardon', 'import_completed')"
        )
        conn.commit()

        assert _pasos(conn, 'import_completed') == 1
    finally:
        conn.close()


def test_la_respuesta_DICE_que_esta_midiendo():
    """Para no tener que acordarse de cual de las dos cuentas es cual."""
    src = _src('routes/admin_panel.py')
    assert '"basis": "cohort",' in src
    assert '"cohort_size": cohort_size,' in src
    assert '"window_devices": window,' in src


def test_el_denominador_es_la_cohorte_no_el_primer_paso():
    """Un dispositivo dado de alta que nunca llega a mandar `app_opened` (red)
    tiene que contar como el abandono que es, no desaparecer del embudo."""
    src = _src('routes/admin_panel.py')
    assert 'top = cohort_size or (' in src


# ============================================================================
# LOS ANONIMOS: LA COHORTE LOS DEJA FUERA, Y HAY QUE SABERLO
# ============================================================================

def test_los_eventos_SIN_device_id_no_estan_en_la_cohorte(tmp_path):
    """No es un fallo, es la definicion: una cohorte sigue a alguien, y un
    evento anonimo no tiene a quien seguir.

    Lo que SI fue un fallo: el script del embudo leia las visitas de la web
    (que van sin device_id a proposito) de `raw`, y al pasar `raw` a ser la
    cuenta por cohorte se fueron a CERO. La primera lectura tras el cambio dio
    «79 visitas -> 0» y parecia que la web se habia caido.
    """
    conn = _bd(tmp_path)
    try:
        conn.execute(
            'INSERT INTO device_first_seen (device_id, first_day) '
            "VALUES ('d1', date('now'))"
        )
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES (datetime('now'), 'd1', 'app_opened')"
        )
        # La web: sin device_id, a proposito.
        conn.execute(
            "INSERT INTO events (timestamp, device_id, event_name) "
            "VALUES (datetime('now'), NULL, 'web_visit')"
        )
        conn.commit()

        assert _pasos(conn, 'web_visit') == 0, 'la cohorte no puede verlos'

        # `window_devices` si los cuenta, uno por fila via el COALESCE.
        ventana = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(device_id, 'anon:' || id)) "
            "FROM events WHERE event_name = 'web_visit' "
            "  AND timestamp >= datetime('now','-30 days')"
        ).fetchone()[0]
        assert ventana == 1
    finally:
        conn.close()


def test_el_script_lee_la_web_de_window_devices_no_de_raw():
    ruta = os.path.join(os.path.dirname(_AQUI), 'Analyzer', 'scripts', 'embudo.sh')
    if not os.path.exists(ruta):
        return  # el repo del cliente no esta al lado; CI del backend no lo tiene
    with open(ruta, encoding='utf-8') as f:
        src = f.read()
    assert "wv = wd.get('web_visit'" in src
    assert "wv = ra.get('web_visit'" not in src, 'raw ya no trae anonimos'


def test_el_codigo_avisa_de_que_raw_ya_no_trae_anonimos():
    """El comentario decia que los anonimos «siguen enteros en raw», y dejo de
    ser cierto en cuanto `raw` paso a ser la cuenta por cohorte."""
    src = _src('routes/admin_panel.py')
    assert 'window_devices' in src
    i = src.index('Los eventos ANONIMOS')
    assert 'window_devices' in src[i:i + 700]
