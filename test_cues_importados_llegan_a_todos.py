"""Los cues que un DJ importa de Rekordbox, Traktor o VirtualDJ llegan a las
zonas de la comunidad.

Auditado el 2026-09-26: no llegaban. El import los guardaba en la biblioteca
del DJ y de ahí no salían: solo subían los cues que se tocaban a mano (uno a
uno, por `/community/cues`) o los de un export. O sea que una biblioteca entera
de hot cues puestos por un DJ de verdad no aportaba nada a las zonas de nadie.

Ahora el cliente los manda por lotes a `/community/cues/batch`. Lo que se ata
aquí:
  - sin aparato registrado, 401 (el `device_id` sale del token, no del cuerpo);
  - por tema hace lo mismo que el envío suelto: SUSTITUYE los cues de ese
    aparato para esa huella, no los acumula;
  - solo se escribe con HUELLA (la trampa de los cues comunitarios);
  - dos DJs que importan el mismo cue forman una zona, y dos aparatos de la
    misma cuenta no (sigue siendo un DJ).

    pytest test_cues_importados_llegan_a_todos.py -v
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from community_cues_endpoint import MAX_TEMAS_POR_LOTE
from main import app, db

client = TestClient(app)

_sembradas = []


@pytest.fixture(autouse=True, scope='module')
def _limpiar_al_salir():
    """La BD es compartida con toda la suite: no dejar rastro."""
    yield
    c = db.conn.cursor()
    for fp in _sembradas:
        c.execute('DELETE FROM community_cues WHERE fingerprint = ?', (fp,))
        c.execute('DELETE FROM community_vote_sources WHERE fingerprint = ?', (fp,))
    db.conn.commit()


def _fp():
    fp = uuid.uuid4().hex
    _sembradas.append(fp)
    return fp


def _aparato(cuenta=None):
    """Registra un aparato (con su cuenta, o en la de [cuenta]) y devuelve
    `(device_id, token)`."""
    import sync_endpoints as se

    conn = se._get_conn()
    device_id, token = uuid.uuid4().hex, 'tok-' + uuid.uuid4().hex
    usuario = cuenta or 'u-' + device_id
    conn.execute('INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)',
                 (usuario, se._now_iso()))
    conn.execute(
        'INSERT INTO user_devices (device_id, user_id, device_type, device_name, '
        "linked_at, device_token) VALUES (?, ?, 'macos', 'Mac', ?, ?)",
        (device_id, usuario, se._now_iso(), token))
    conn.commit()
    return device_id, token, usuario


def _cue(ms, tipo='hotCue'):
    return {'type': tipo, 'position_ms': ms}


def _enviar(token, items, ip='10.0.0.1'):
    return client.post('/community/cues/batch', json={'items': items},
                       headers={'X-Device-Token': token,
                                'X-Forwarded-For': ip})


def _mis_cues(fp, device):
    return sorted(r['position_ms'] for r in db.conn.execute(
        'SELECT position_ms FROM community_cues '
        'WHERE fingerprint = ? AND device_id = ?', (fp, device)).fetchall())


def test_sin_aparato_registrado_401():
    cuerpo = {'items': [{'fingerprint': _fp(), 'cues': [_cue(1000)]}]}
    assert client.post('/community/cues/batch', json=cuerpo).status_code == 401
    r = client.post('/community/cues/batch', json=cuerpo,
                    headers={'X-Device-Token': 'tok-inventado-' + 'x' * 20})
    assert r.status_code == 401


def test_guarda_los_cues_del_aparato_del_token():
    dev, tok, _ = _aparato()
    fp1, fp2 = _fp(), _fp()
    r = _enviar(tok, [
        {'fingerprint': fp1, 'cues': [_cue(1000), _cue(30000)]},
        {'fingerprint': fp2, 'cues': [_cue(5000, 'loop')]},
    ])
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'temas': 2, 'cues': 3, 'descartados': 0}
    assert _mis_cues(fp1, dev) == [1000, 30000]
    assert _mis_cues(fp2, dev) == [5000]


def test_reenviar_sustituye_no_acumula():
    # Es lo que hace el envío suelto, y lo que permite reenviar la biblioteca
    # entera sin duplicar nada.
    dev, tok, _ = _aparato()
    fp = _fp()
    _enviar(tok, [{'fingerprint': fp, 'cues': [_cue(1000), _cue(2000)]}])
    _enviar(tok, [{'fingerprint': fp, 'cues': [_cue(9000)]}])
    assert _mis_cues(fp, dev) == [9000]


def test_solo_con_huella():
    _, tok, _ = _aparato()
    r = _enviar(tok, [
        {'fingerprint': 'imp_' + 'a' * 28, 'cues': [_cue(1000)]},   # ghost
        {'fingerprint': 'no-es-una-huella', 'cues': [_cue(1000)]},
        {'fingerprint': _fp(), 'cues': [_cue(1000)]},
    ])
    assert r.json()['temas'] == 1
    assert r.json()['descartados'] == 2


def test_la_clave_vieja_de_este_aparato_se_limpia():
    dev, tok, _ = _aparato()
    fp, vieja = _fp(), _fp()
    db.conn.execute(
        'INSERT INTO community_cues (fingerprint, device_id, cue_type, '
        "position_ms, created_at) VALUES (?, ?, 'hotCue', 1000, 'x')",
        (vieja, dev))
    db.conn.commit()
    _enviar(tok, [{'fingerprint': fp, 'cues': [_cue(1000)], 'legacy_key': vieja}])
    assert _mis_cues(vieja, dev) == []
    assert _mis_cues(fp, dev) == [1000]


def test_un_lote_demasiado_grande_se_rechaza():
    _, tok, _ = _aparato()
    items = [{'fingerprint': uuid.uuid4().hex, 'cues': [_cue(1000)]}
             for _ in range(MAX_TEMAS_POR_LOTE + 1)]
    assert _enviar(tok, items).status_code == 400


def test_dos_djs_que_importan_el_mismo_cue_hacen_una_zona():
    fp = _fp()
    _, tok_a, _ = _aparato()
    _, tok_b, _ = _aparato()
    _enviar(tok_a, [{'fingerprint': fp, 'cues': [_cue(64000)]}], ip='10.0.1.1')
    assert client.get(f'/community/cues/{fp}').json()['zones'] == [], \
        'con un solo DJ no hay zona'
    _enviar(tok_b, [{'fingerprint': fp, 'cues': [_cue(64200)]}], ip='10.0.1.2')
    zonas = client.get(f'/community/cues/{fp}').json()['zones']
    assert len(zonas) == 1
    assert zonas[0]['dj_count'] == 2


def test_dos_aparatos_de_la_misma_cuenta_son_un_dj():
    # El escritorio y el móvil del mismo DJ tienen los mismos cues (viajan por
    # sync) y los dos los mandan: eso no puede fabricar una zona.
    fp = _fp()
    _, tok_mac, cuenta = _aparato()
    _, tok_movil, _ = _aparato(cuenta=cuenta)
    _enviar(tok_mac, [{'fingerprint': fp, 'cues': [_cue(64000)]}], ip='10.0.2.1')
    _enviar(tok_movil, [{'fingerprint': fp, 'cues': [_cue(64000)]}], ip='10.0.2.2')
    assert client.get(f'/community/cues/{fp}').json()['zones'] == []


def test_sin_cues_retira_lo_que_aportaba_este_aparato():
    # Borrar todos los cues de un tema retira tu aportación a las zonas; hasta
    # el 2026-09-26 una lista vacía era un error y los viejos seguían contando.
    dev, tok, _ = _aparato()
    otro, tok_otro, _ = _aparato()
    fp = _fp()
    _enviar(tok, [{'fingerprint': fp, 'cues': [_cue(64000)]}], ip='10.0.3.1')
    _enviar(tok_otro, [{'fingerprint': fp, 'cues': [_cue(64000)]}], ip='10.0.3.2')
    assert len(client.get(f'/community/cues/{fp}').json()['zones']) == 1
    r = _enviar(tok, [{'fingerprint': fp, 'cues': []}], ip='10.0.3.1')
    assert r.json()['temas'] == 1
    assert _mis_cues(fp, dev) == []
    assert _mis_cues(fp, otro) == [64000], 'lo de los demás no se toca'
    assert client.get(f'/community/cues/{fp}').json()['zones'] == []


def test_el_envio_suelto_sin_cues_tambien_retira():
    dev = 'cuesvacios_' + uuid.uuid4().hex[:8]
    fp = _fp()
    client.post('/community/cues', json={'fingerprint': fp, 'device_id': dev,
                                         'cues': [_cue(1000)]})
    r = client.post('/community/cues', json={'fingerprint': fp, 'device_id': dev,
                                             'cues': []})
    assert r.json()['status'] == 'ok'
    assert _mis_cues(fp, dev) == []
