"""
El token de ShazamKit para Android (2026-09-30).

En iPhone, ShazamKit se autoriza solo: el App ID lleva el servicio ShazamKit
marcado y el sistema lo sabe. En Android no hay nada de eso: el SDK pide un
«developer token», un JWT firmado (ES256) con una clave de Apple Developer
de tipo Media Services con ShazamKit. La clave NO puede ir en la app (se
sacaría del APK y valdría para cualquiera), así que se firma aquí y el móvil
lo pide con `GET /shazam/token`.

Variables de Render (Environment):
  SHAZAM_TEAM_ID      Team ID de la cuenta de Apple Developer (10 caracteres)
  SHAZAM_KEY_ID       Key ID de la clave Media Services (10 caracteres)
  SHAZAM_PRIVATE_KEY  el contenido del .p8, entero (-----BEGIN PRIVATE KEY-----
                      …). Vale con saltos de línea de verdad o con «\\n».

Sin las tres, el endpoint da 503 y el Android sigue con AudD solo.

El token dura [DURACION] (Apple admite hasta 6 meses) y se reutiliza hasta
que le queda [RENOVAR]: firmar cuesta poco, pero no hay por qué hacerlo en
cada petición. Rotar la CLAVE (no el token) es cosa de Apple Developer: si se
revoca, se cambian las variables y listo.
"""
import base64
import json
import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DURACION = 30 * 24 * 3600
RENOVAR = 7 * 24 * 3600

_cache = {'token': None, 'expira': 0, 'huella': None}
_cerrojo = threading.Lock()


def _b64url(datos: bytes) -> str:
    return base64.urlsafe_b64encode(datos).rstrip(b'=').decode('ascii')


def _clave_pem() -> Optional[str]:
    pem = (os.environ.get('SHAZAM_PRIVATE_KEY') or '').strip()
    if not pem:
        return None
    # Pegada en una sola línea en el panel de Render: los saltos llegan como
    # «\n» literales.
    if '\\n' in pem and '\n' not in pem:
        pem = pem.replace('\\n', '\n')
    return pem


def configurado() -> bool:
    return bool((os.environ.get('SHAZAM_TEAM_ID') or '').strip()
                and (os.environ.get('SHAZAM_KEY_ID') or '').strip()
                and _clave_pem())


def firmar(team_id: str, key_id: str, pem: str, ahora: int,
           duracion: int = DURACION) -> str:
    """El JWT: cabecera {alg: ES256, kid}, cuerpo {iss, iat, exp}, y la firma
    ECDSA P-256 / SHA-256 en crudo (r‖s, 64 bytes), que es lo que pide JWS;
    `cryptography` la da en DER."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import \
        decode_dss_signature

    clave = serialization.load_pem_private_key(pem.encode('utf-8'),
                                               password=None)
    if not isinstance(clave, ec.EllipticCurvePrivateKey):
        raise ValueError('la clave no es de curva elíptica (¿es el .p8?)')
    cabecera = {'alg': 'ES256', 'kid': key_id}
    cuerpo = {'iss': team_id, 'iat': int(ahora), 'exp': int(ahora + duracion)}
    a_firmar = (_b64url(json.dumps(cabecera, separators=(',', ':')).encode())
                + '.'
                + _b64url(json.dumps(cuerpo, separators=(',', ':')).encode()))
    der = clave.sign(a_firmar.encode('ascii'), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    return a_firmar + '.' + _b64url(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))


def token_de_shazam(reloj: Callable[[], float] = time.time) -> Optional[dict]:
    """{'token', 'expira'} (epoch en segundos), o None si no está configurado
    o la clave no sirve (queda en el log)."""
    if not configurado():
        return None
    team = os.environ['SHAZAM_TEAM_ID'].strip()
    kid = os.environ['SHAZAM_KEY_ID'].strip()
    pem = _clave_pem()
    huella = (team, kid, hash(pem))
    ahora = int(reloj())
    with _cerrojo:
        if (_cache['token'] and _cache['huella'] == huella
                and _cache['expira'] - ahora > RENOVAR):
            return {'token': _cache['token'], 'expira': _cache['expira']}
        try:
            token = firmar(team, kid, pem, ahora)
        except Exception as e:  # noqa: BLE001 - una clave mal pegada no tumba nada
            logger.warning(f"[Shazam] no se pudo firmar el token: {e}")
            return None
        _cache.update(token=token, expira=ahora + DURACION, huella=huella)
        logger.info(f"[Shazam] token nuevo (kid={kid}, "
                    f"{DURACION // 86400} días)")
        return {'token': token, 'expira': ahora + DURACION}
