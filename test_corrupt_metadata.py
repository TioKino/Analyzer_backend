"""
Tests de robustez para la extraccion de metadata/artwork ante ficheros
CORRUPTOS o TRUNCADOS del cliente (tipico de una descarga de nube incompleta
en movil: iCloud/Dropbox/Drive).

Reproduce los errores reales vistos en logs de produccion:
  - MP4MetadataError: unpack requires a buffer of 4 bytes
  - MP4StreamInfoError: not a MP4 file  /  KeyError: b'moov' not found
  - TypeError: argument of type 'NoneType' is not iterable  (audio.tags is None)

Ninguno debe propagarse: extract_artwork_from_file / extract_id3_metadata son AUXILIARES
(best-effort) y no pueden tumbar el analisis ni ensuciar la telemetria de
errores del backend. El fallback del /analyze ya devuelve un resultado
degradado (200) con la metadata del filename.
"""
import os
import tempfile

import pytest

from artwork_and_cuepoints import extract_artwork_from_file, extract_id3_metadata


def _write_tmp(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path


# Bytes basura con extension de audio: mutagen intenta parsear y lanza sus
# propios errores (MP4MetadataError / MP4StreamInfoError / ID3 / FLAC), que NO
# son subclase de IOError/ValueError -> antes escapaban.
GARBAGE = b'\x00\x01\x02\x03not a real audio file at all' * 8

# Cabecera 'ftyp' minima para que mutagen lo tome por MP4 pero sin 'moov'
# (-> "not a MP4 file" / moov not found), simulando un .m4a truncado.
TRUNCATED_MP4 = (
    b'\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00M4A mp42isom'
    b'\x00\x00\x00\x08free'
)


@pytest.mark.parametrize('suffix', ['.m4a', '.mp4', '.mp3', '.flac', '.aac'])
def test_extract_artwork_no_revienta_con_basura(suffix):
    path = _write_tmp(suffix, GARBAGE)
    try:
        assert extract_artwork_from_file(path) is None  # no artwork, sin excepcion
    finally:
        os.remove(path)


@pytest.mark.parametrize('suffix', ['.m4a', '.mp4', '.mp3', '.flac'])
def test_extract_id3_no_revienta_con_basura(suffix):
    path = _write_tmp(suffix, GARBAGE)
    try:
        meta = extract_id3_metadata(path)
        # Devuelve el dict con todos los campos None (nada legible), sin lanzar.
        assert isinstance(meta, dict)
        assert meta['title'] is None
        assert meta['artist'] is None
    finally:
        os.remove(path)


def test_extract_artwork_mp4_truncado_no_revienta():
    path = _write_tmp('.m4a', TRUNCATED_MP4)
    try:
        assert extract_artwork_from_file(path) is None
    finally:
        os.remove(path)


def test_extract_id3_mp4_truncado_no_revienta():
    path = _write_tmp('.m4a', TRUNCATED_MP4)
    try:
        meta = extract_id3_metadata(path)
        assert isinstance(meta, dict)
    finally:
        os.remove(path)


def test_artwork_tags_none_no_lanza_typeerror(monkeypatch):
    """El caso exacto del log: audio.tags is None en un MP4 -> 'covr' in None
    lanzaba TypeError. Se stubbea mutagen.File para forzar ese estado."""
    import artwork_and_cuepoints as mod
    from mutagen.mp4 import MP4

    class _FakeMP4(MP4):
        def __init__(self):  # no llamamos al parser real
            self.tags = None

    fake = _FakeMP4()
    # extract_artwork_from_file hace `from mutagen import File as MutagenFile`; parcheamos
    # el simbolo en el modulo mutagen para que devuelva nuestro fake.
    import mutagen
    monkeypatch.setattr(mutagen, 'File', lambda *a, **k: fake)

    path = _write_tmp('.m4a', TRUNCATED_MP4)
    try:
        # No debe lanzar TypeError; devuelve None (sin artwork).
        assert mod.extract_artwork_from_file(path) is None
    finally:
        os.remove(path)
