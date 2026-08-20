"""Auth POR DISPOSITIVO en /sync — fase 1 de 3.

Por que existe el mecanismo: SYNC_AUTH_SECRET viaja compilado en el binario de
cada cliente (--dart-define), asi que cualquiera lo saca con `strings` del
.app/.exe/.apk. No es un secreto filtrado: es que un secreto compartido dentro
de una app distribuida NO PUEDE serlo. Y hoy ese HMAC es la UNICA puerta —
_require_user_id solo comprueba que el device_id este REGISTRADO, no que quien
llama SEA ese dispositivo. Con el secreto y un device_id conocido se puede leer
y sobrescribir la biblioteca de cualquier usuario.

Lo que cubre esta fase (y lo que NO):
  - El token se emite UNA sola vez por dispositivo y no se vuelve a revelar.
  - Si el cliente manda X-Device-Token, tiene que ser el suyo.
  - Si NO lo manda, se acepta igual que siempre (los dispositivos en campo no
    se rompen). O sea: quien omita la cabecera sigue entrando. Cerrar eso es
    la fase 3 y depende de que sync_auth_stats demuestre adopcion total.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sync_endpoints as se  # noqa: E402


@pytest.fixture
def conn(monkeypatch):
    """sync.db temporal con el schema real (incluye la migracion del token)."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    monkeypatch.setattr(se, '_DB_PATH', path)
    monkeypatch.setattr(se, '_conn', None)
    c = se._get_conn()
    c.row_factory = sqlite3.Row
    yield c
    monkeypatch.setattr(se, '_conn', None)
    try:
        c.close()
        os.unlink(path)
    except OSError:
        pass


def _register(conn, device_id, user_id='u1'):
    conn.execute("INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
                 (user_id, se._now_iso()))
    conn.execute(
        "INSERT INTO user_devices (device_id, user_id, device_type, device_name, linked_at) "
        "VALUES (?, ?, 'macos', 'Mac', ?)",
        (device_id, user_id, se._now_iso()),
    )
    conn.commit()


class TestEmisionDelToken:

    def test_se_emite_una_vez_y_no_se_revela_mas(self, conn):
        """Lo importante: si register devolviera el token siempre, cualquiera
        con el secreto global podria pedir el de un device_id conocido y
        quedarse con la cuenta — justo lo que esto viene a impedir."""
        _register(conn, 'dja_a')

        primero = se._issue_device_token(conn, 'dja_a')
        assert primero, "no emitio token la primera vez"

        segundo = se._issue_device_token(conn, 'dja_a')
        assert segundo is None, "revelo el token una segunda vez"

    def test_el_token_persiste_y_es_el_mismo(self, conn):
        _register(conn, 'dja_a')
        emitido = se._issue_device_token(conn, 'dja_a')
        row = conn.execute(
            "SELECT device_token FROM user_devices WHERE device_id = 'dja_a'"
        ).fetchone()
        assert row['device_token'] == emitido

    def test_tokens_distintos_por_dispositivo(self, conn):
        _register(conn, 'dja_a')
        _register(conn, 'dja_b', user_id='u2')
        assert se._issue_device_token(conn, 'dja_a') != se._issue_device_token(conn, 'dja_b')

    def test_token_con_entropia_de_verdad(self, conn):
        """A diferencia del device_id, que es un timestamp en base36 sin una
        sola pizca de aleatoriedad."""
        _register(conn, 'dja_a')
        t = se._issue_device_token(conn, 'dja_a')
        assert len(t) >= 32

    def test_dispositivo_inexistente_no_emite(self, conn):
        assert se._issue_device_token(conn, 'dja_fantasma') is None


class TestExtraccionDelDeviceId:
    """La verificacion vive en la dependency, que es el unico sitio con acceso
    al Request; de ahi que tenga que sacar el device_id del path o del body."""

    class _Req:
        def __init__(self, path_params=None):
            self.path_params = path_params or {}

    def test_lo_saca_del_path(self):
        r = self._Req({'device_id': 'dja_path'})
        assert se._device_id_from_request(r, b'') == 'dja_path'

    def test_lo_saca_del_body(self):
        r = self._Req()
        assert se._device_id_from_request(r, b'{"device_id": "dja_body"}') == 'dja_body'

    def test_el_path_gana_al_body(self):
        r = self._Req({'device_id': 'dja_path'})
        out = se._device_id_from_request(r, b'{"device_id": "dja_body"}')
        assert out == 'dja_path'

    def test_body_no_json_no_explota(self):
        assert se._device_id_from_request(self._Req(), b'NOT JSON') is None

    def test_body_sin_device_id(self):
        assert se._device_id_from_request(self._Req(), b'{"otra": 1}') is None

    def test_body_json_que_no_es_dict(self):
        assert se._device_id_from_request(self._Req(), b'[1,2,3]') is None


class TestContadoresDeAdopcion:
    """sync_auth_stats es lo que permite decidir con datos: (1) retirar el
    secreto viejo cuando nadie lo use, (2) exigir el token cuando todos lo
    manden."""

    def test_acumula_por_slot_y_token(self, conn):
        se._record_sync_auth(0, True)
        se._record_sync_auth(0, True)
        se._record_sync_auth(0, False)
        se._record_sync_auth(1, False)

        rows = {
            (r['secret_slot'], r['has_token']): r['n']
            for r in conn.execute(
                "SELECT secret_slot, has_token, n FROM sync_auth_stats")
        }
        assert rows[(0, 1)] == 2
        assert rows[(0, 0)] == 1
        assert rows[(1, 0)] == 1

    def test_es_best_effort_y_nunca_lanza(self, conn):
        conn.execute("DROP TABLE sync_auth_stats")
        conn.commit()
        se._record_sync_auth(0, True)  # no debe propagar excepcion


class TestNoRompeALosQueYaEstan:

    def test_dispositivo_sin_token_sigue_registrado(self, conn):
        """Los dispositivos en campo no tienen token. La columna nace NULL y
        eso no puede impedirles resolver su user_id."""
        _register(conn, 'dja_viejo')
        row = conn.execute(
            "SELECT device_token FROM user_devices WHERE device_id = 'dja_viejo'"
        ).fetchone()
        assert row['device_token'] is None
        assert se._require_user_id(conn, 'dja_viejo') == 'u1'

    def test_la_migracion_es_idempotente(self, conn):
        """_init_tables corre en cada arranque; el ALTER no puede petar."""
        se._init_tables(conn)
        se._init_tables(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(user_devices)")]
        assert cols.count('device_token') == 1


class TestVinculacionEntreDispositivos:
    """El codigo de vinculacion y el device_token son cosas DISTINTAS:

      - codigo de vinculacion: 6 caracteres, 10 min, lo teclea el usuario. Dice
        A QUE USUARIO pertenece un dispositivo. Es una operacion de cuenta.
      - device_token: credencial permanente que acredita QUE APARATO eres. El
        usuario no lo ve ni lo teclea.

    Se cruzan en un punto: /sync/link/join hace DELETE + INSERT sobre
    user_devices, y eso borraba el token del dispositivo que se une. En fase 1
    no se notaria (el token solo se valida si viene y la fila quedaba a NULL),
    pero en fase 3 seria dejar a ese dispositivo fuera de sus propios datos.
    """

    def _link_code(self, conn, code, user_id):
        from datetime import datetime, timedelta, timezone
        exp = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        conn.execute(
            "INSERT INTO link_codes (code, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (code, user_id, se._now_iso(), exp),
        )
        conn.commit()

    def test_el_token_sobrevive_a_vincular_con_otro_usuario(self, conn):
        _register(conn, 'dja_movil', user_id='u_movil')
        token = se._issue_device_token(conn, 'dja_movil')
        conn.execute("INSERT OR IGNORE INTO users (user_id, created_at) VALUES ('u_pc', ?)",
                     (se._now_iso(),))
        conn.commit()
        self._link_code(conn, 'ABC123', 'u_pc')

        import asyncio

        class _Req:
            device_id = 'dja_movil'
            code = 'ABC123'
            device_type = 'ios'
            device_name = 'iPhone'

        class _Client:
            host = '127.0.0.1'

        class _HttpReq:
            client = _Client()
            headers: dict = {}

        asyncio.run(se.sync_link_join(_Req(), _HttpReq()))

        row = conn.execute(
            "SELECT user_id, device_token FROM user_devices WHERE device_id = 'dja_movil'"
        ).fetchone()
        assert row['user_id'] == 'u_pc', "no cambio de usuario"
        assert row['device_token'] == token, "la vinculacion le borro el token"

    def test_vincular_no_emite_un_token_nuevo(self, conn):
        """El aparato es el mismo: su credencial no debe rotar por cambiar de
        cuenta (si rotara, el cliente se quedaria con una copia obsoleta)."""
        _register(conn, 'dja_movil', user_id='u_movil')
        token = se._issue_device_token(conn, 'dja_movil')
        assert se._issue_device_token(conn, 'dja_movil') is None
        row = conn.execute(
            "SELECT device_token FROM user_devices WHERE device_id = 'dja_movil'"
        ).fetchone()
        assert row['device_token'] == token


# ══════════════════════════════════════════════════════════════════════
# FASE 2.5 — el token se EXIGE por dispositivo, sin dia D
# ══════════════════════════════════════════════════════════════════════
#
# La fase 1 solo validaba el token SI venia, asi que quien omitia la cabecera
# seguia entrando: el agujero quedaba abierto para todos.
#
# El plan escrito decia "si ese device_id ya tiene token EMITIDO, exigelo". Esa
# premisa es FALSA y aplicarla habria provocado una caida: `_issue_device_token`
# se llama en TODO /sync/register sin mirar la version del cliente, asi que un
# cliente <= 2.9.8 recibe el token, lo IGNORA (su codigo no conoce el campo) y
# deja `device_token` guardado en el servidor. Exigirlo por "emitido" habria
# echado de su propia biblioteca a todo ese parque.
#
# Lo que SI demuestra que un aparato sabe mandar el token es haberlo mandado.
# De ahi `token_seen_at`: se sella al primer token correcto y a partir de ahi se
# exige. Cada usuario queda protegido al actualizar, sin esperar a nadie.
import asyncio  # noqa: E402


class _FakeRequest:
    """Request minimo: _verify_sync_auth solo usa headers y body()."""

    def __init__(self, headers, body=b'{}'):
        self.headers = headers
        self._body = body

    async def body(self):
        return self._body


def _verify(headers, body=b'{}'):
    return asyncio.get_event_loop().run_until_complete(
        se._verify_sync_auth(_FakeRequest(headers, body))
    )


@pytest.fixture
def sin_secreto(monkeypatch):
    """Dev mode local: SYNC_AUTH_SECRET vacio -> _verify_sync_auth sale antes
    de mirar el token. Para probar la parte del token hace falta secreto."""
    monkeypatch.setattr(se, 'SYNC_AUTH_SECRET', '')
    monkeypatch.delenv('RENDER', raising=False)
    monkeypatch.delenv('RAILWAY_ENVIRONMENT', raising=False)


class TestFase25Enforcement:
    """Comportamiento del sellado + exigencia. Se prueba sobre la BD, que es
    donde vive la decision, sin montar el HMAC completo."""

    def test_la_columna_nace_null(self, conn):
        _register(conn, 'dja_x')
        row = conn.execute(
            "SELECT token_seen_at FROM user_devices WHERE device_id='dja_x'"
        ).fetchone()
        assert row['token_seen_at'] is None

    def test_migracion_idempotente(self, conn):
        se._init_tables(conn)
        se._init_tables(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(user_devices)")]
        assert cols.count('token_seen_at') == 1

    def test_token_emitido_NO_basta_para_exigir(self, conn):
        """EL CASO QUE HABRIA TUMBADO PRODUCCION. Un cliente viejo recibe token
        en /sync/register y lo ignora: la fila tiene device_token pero el
        aparato no sabe mandarlo. NO puede quedar marcado como protegido."""
        _register(conn, 'dja_viejo')
        emitido = se._issue_device_token(conn, 'dja_viejo')
        assert emitido, 'register emite token a cualquiera, tambien a clientes viejos'
        row = conn.execute(
            "SELECT device_token, token_seen_at FROM user_devices "
            "WHERE device_id='dja_viejo'"
        ).fetchone()
        assert row['device_token'] is not None   # emitido
        assert row['token_seen_at'] is None      # pero NO exigible

    def test_sellado_marca_el_dispositivo(self, conn):
        _register(conn, 'dja_nuevo')
        se._issue_device_token(conn, 'dja_nuevo')
        conn.execute(
            "UPDATE user_devices SET token_seen_at = ? "
            "WHERE device_id = ? AND token_seen_at IS NULL",
            (se._now_iso(), 'dja_nuevo'),
        )
        conn.commit()
        row = conn.execute(
            "SELECT token_seen_at FROM user_devices WHERE device_id='dja_nuevo'"
        ).fetchone()
        assert row['token_seen_at'] is not None

    def test_el_sellado_no_se_reescribe(self, conn):
        """El UPDATE lleva `AND token_seen_at IS NULL` para no escribir en cada
        peticion de sync: sellar es una vez en la vida del dispositivo."""
        _register(conn, 'dja_s')
        se._issue_device_token(conn, 'dja_s')
        primero = '2020-01-01T00:00:00+00:00'
        conn.execute(
            "UPDATE user_devices SET token_seen_at = ? "
            "WHERE device_id = ? AND token_seen_at IS NULL",
            (primero, 'dja_s'),
        )
        conn.execute(
            "UPDATE user_devices SET token_seen_at = ? "
            "WHERE device_id = ? AND token_seen_at IS NULL",
            (se._now_iso(), 'dja_s'),
        )
        conn.commit()
        row = conn.execute(
            "SELECT token_seen_at FROM user_devices WHERE device_id='dja_s'"
        ).fetchone()
        assert row['token_seen_at'] == primero

    def test_vincular_no_desprotege_al_dispositivo(self, conn):
        """/sync/link/join hace DELETE+INSERT sobre user_devices y solo
        arrastraba `device_token`. Si pierde `token_seen_at`, un dispositivo YA
        PROTEGIDO vuelve a aceptar peticiones sin token: vincular reabriria el
        agujero justo en el aparato que ya estaba cerrado, y en silencio.

        El test LLAMA al endpoint de verdad: comprobarlo con un UPDATE a mano
        pasaria sin ejercitar el DELETE+INSERT, que es donde esta el fallo."""
        _register(conn, 'dja_p', user_id='u_a')
        tok = se._issue_device_token(conn, 'dja_p')
        sellado = se._now_iso()
        conn.execute(
            "UPDATE user_devices SET token_seen_at = ? WHERE device_id = ?",
            (sellado, 'dja_p'),
        )
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at) VALUES ('u_b', ?)",
            (se._now_iso(),))
        conn.commit()

        from datetime import datetime, timedelta, timezone
        exp = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        conn.execute(
            "INSERT INTO link_codes (code, user_id, created_at, expires_at) "
            "VALUES ('ZZZ999', 'u_b', ?, ?)", (se._now_iso(), exp))
        conn.commit()

        class _Req:
            device_id = 'dja_p'
            code = 'ZZZ999'
            device_type = 'macos'
            device_name = 'Mac'

        class _Client:
            host = '127.0.0.1'

        class _HttpReq:
            client = _Client()
            headers: dict = {}

        asyncio.run(se.sync_link_join(_Req(), _HttpReq()))

        row = conn.execute(
            "SELECT user_id, device_token, token_seen_at FROM user_devices "
            "WHERE device_id='dja_p'"
        ).fetchone()
        assert row['user_id'] == 'u_b', 'no cambio de usuario'
        assert row['device_token'] == tok, 'la vinculacion le borro el token'
        assert row['token_seen_at'] == sellado, \
            'la vinculacion DESPROTEGIO el dispositivo'
