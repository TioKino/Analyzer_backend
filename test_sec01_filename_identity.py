"""
SEC-01 — el nombre de fichero NO es identidad.

La tabla `tracks` de Render es COMPARTIDA por todos los usuarios, y `/analyze`
casaba por `filename` a secas ANTES de calcular ninguna huella. Con nombres
universales en bibliotecas DJ (`01.mp3`, `track01.mp3`, `Untitled.mp3`) eso
producia dos fallos, los dos silenciosos:

  1. Usuario B sube su '01 - Intro.mp3' y recibe el analisis del de A.
  2. Usuario B reanaliza con force=true y BORRA el track de A
     (`DELETE FROM tracks WHERE filename = ?`, sin filtro de propietario).

Ninguno de los dos lanza error: B se va con datos de otra persona creyendo que
son suyos, y A pierde su analisis sin enterarse. Ademas contradecia de raiz la
promesa del producto —"agrupamos por sonido"— porque el camino mas corto del
codigo agrupaba por nombre.

Estos tests van contra la CAPA DE BD y el helper de identidad, no contra el
endpoint completo: /analyze necesita audio real y librosa, y lo que hay que
fijar aqui es la regla de identidad, no el DSP.
"""

import tempfile
import uuid

import pytest


@pytest.fixture()
def db():
    from database import AnalysisDB

    return AnalysisDB(tempfile.mktemp(suffix=".db"))


def _guardar(db, *, filename, fingerprint, artist, bpm=128.0, id_=None):
    """`save_track` exige id/filename/duration/bpm/energy_dj/genre/track_type
    sin default, asi que el payload va completo."""
    db.save_track({
        'id': id_ or fingerprint,
        'filename': filename,
        'artist': artist,
        'title': 'Tema',
        'duration': 300.0,
        'bpm': bpm,
        'key': 'Am',
        'camelot': '8A',
        'energy_dj': 7,
        'genre': 'Techno',
        'track_type': 'peak_time',
        'fingerprint': fingerprint,
    })


class TestDosUsuariosMismoNombre:
    """El escenario exacto del informe."""

    NOMBRE = "01 - Intro.mp3"

    def test_la_huella_distingue_lo_que_el_nombre_confunde(self, db):
        fp_a, fp_b = uuid.uuid4().hex, uuid.uuid4().hex
        _guardar(db, filename=self.NOMBRE, fingerprint=fp_a,
                 artist="DJ ALPHA", bpm=128.0)

        # El lookup por NOMBRE encuentra el track de A aunque B suba otro audio.
        por_nombre = db.get_track_by_filename(self.NOMBRE)
        assert por_nombre is not None, "el atajo por nombre sigue existiendo"

        # Pero la huella de B NO coincide -> el cache-hit debe descartarse.
        guardada = (db._row_to_dict(por_nombre) or {}).get('fingerprint')
        assert guardada == fp_a
        assert guardada != fp_b, (
            "si estas dos huellas fueran iguales el test no probaria nada"
        )

    def test_el_lookup_por_huella_no_devuelve_el_track_ajeno(self, db):
        """La comprobacion que ahora hace /analyze: por huella, no por nombre."""
        fp_a, fp_b = uuid.uuid4().hex, uuid.uuid4().hex
        _guardar(db, filename=self.NOMBRE, fingerprint=fp_a, artist="DJ ALPHA")

        assert db.get_track_by_fingerprint(fp_b) is None, (
            "el audio de B no esta en la BD; devolver algo aqui seria servirle "
            "el analisis de otra persona"
        )
        mio = db.get_track_by_fingerprint(fp_a)
        assert mio and mio['artist'] == "DJ ALPHA"

    def test_el_mismo_audio_con_otro_nombre_SI_casa(self, db):
        """El reverso: no romper el dedup real. El mismo audio subido desde el
        movil con otro nombre debe seguir encontrandose."""
        fp = uuid.uuid4().hex
        _guardar(db, filename="pista_movil.mp3", fingerprint=fp, artist="DJ MIO")

        encontrado = db.get_track_by_fingerprint(fp)
        assert encontrado and encontrado['artist'] == "DJ MIO", (
            "el dedup por sonido entre dispositivos no debe romperse"
        )


class TestBorradoConForce:
    """`force=true` borraba por nombre, sin filtro de propietario."""

    def test_borrar_por_id_solo_toca_una_fila(self, db):
        nombre = "track01.mp3"
        fp_a, fp_b = uuid.uuid4().hex, uuid.uuid4().hex
        _guardar(db, filename=nombre, fingerprint=fp_a, artist="DJ ALPHA")
        _guardar(db, filename=nombre, fingerprint=fp_b, artist="DJ BETA")

        # B reanaliza SU audio: se borra su fila y la de A sobrevive.
        objetivo = db.get_track_by_fingerprint(fp_b)
        db.delete_track(objetivo['id'])

        assert db.get_track_by_fingerprint(fp_b) is None, "no se borro el propio"
        superviviente = db.get_track_by_fingerprint(fp_a)
        assert superviviente is not None, (
            "el track de OTRO usuario con el mismo nombre fue borrado: "
            "es exactamente el fallo de SEC-01"
        )
        assert superviviente['artist'] == "DJ ALPHA"

    def test_el_borrado_por_nombre_arrasa_con_todo(self, db):
        """Deja constancia de POR QUE no se usa `delete_track_by_filename` en
        el camino de force. El metodo sigue existiendo (lo usan otros flujos),
        pero sobre un nombre compartido se lleva las filas de todos."""
        nombre = "Untitled.mp3"
        fp_a, fp_b = uuid.uuid4().hex, uuid.uuid4().hex
        _guardar(db, filename=nombre, fingerprint=fp_a, artist="DJ ALPHA")
        _guardar(db, filename=nombre, fingerprint=fp_b, artist="DJ BETA")

        db.delete_track_by_filename(nombre)

        assert db.get_track_by_fingerprint(fp_a) is None
        assert db.get_track_by_fingerprint(fp_b) is None
        # Si algun dia este test falla porque solo borra una, genial: querra
        # decir que el metodo se acoto y el comentario de arriba sobra.


class TestElEndpointDeVerdad:
    """Los tests de arriba fijan la REGLA en la capa de BD. Estos prueban que
    `/analyze` la aplica, que es lo que de verdad importa: la regla podia estar
    bien y el endpoint seguir usando el atajo por nombre."""

    NOMBRE = "01 - Intro.mp3"

    @pytest.fixture(scope="class")
    def app_mod(self):
        import main

        main.GENRE_DETECTOR_ENABLED = False
        main.ARTWORK_ENABLED = False
        main.AUDD_AUTO_ENABLED = False
        return main

    @pytest.fixture(scope="class")
    def client(self, app_mod):
        from fastapi.testclient import TestClient

        return TestClient(app_mod.app)

    @staticmethod
    def _wav_bytes(freq=440.0, secs=6.0, sr=22050):
        """WAV sintetico minimo. Distintas `freq` -> distintos bytes -> distinta
        huella MD5, que es lo que se quiere contrastar."""
        import io

        import numpy as np
        import soundfile as sf

        t = np.arange(int(sr * secs)) / sr
        y = (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue()

    def test_no_devuelve_el_analisis_de_otro_usuario(self, client, app_mod):
        """EL TEST QUE IMPORTA. A tiene '01 - Intro.mp3' en la BD colectiva; B
        sube OTRO audio con el mismo nombre. B no puede recibir los datos de A."""
        audio_de_B = self._wav_bytes(freq=440.0)

        _guardar(app_mod.db, filename=self.NOMBRE,
                 fingerprint=uuid.uuid4().hex,  # huella de OTRO audio
                 artist="DJ ALPHA", bpm=128.0)

        r = client.post("/analyze", files={
            "file": (self.NOMBRE, audio_de_B, "audio/wav")
        })
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("artist") != "DJ ALPHA", (
            "SEC-01 VIVO: /analyze devolvio el analisis de otro usuario solo "
            "porque el fichero se llamaba igual"
        )

    def test_el_mismo_audio_si_reusa_el_cache(self, client, app_mod):
        """El reverso: no romper el cache legitimo. Si la huella coincide, el
        cache-hit debe seguir funcionando — si no, cada subida reanalizaria."""
        import hashlib

        audio = self._wav_bytes(freq=330.0)
        fp = hashlib.md5(audio).hexdigest()
        _guardar(app_mod.db, filename="mio.mp3", fingerprint=fp,
                 artist="DJ MIO", bpm=123.0)

        r = client.post("/analyze", files={
            "file": ("mio.mp3", audio, "audio/wav")
        })
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("artist") == "DJ MIO" and d.get("bpm") == 123.0, (
            "el cache-hit por huella coincidente dejo de funcionar; cada "
            "subida repetida volveria a analizar"
        )

    def test_force_no_borra_el_track_de_otro(self, client, app_mod):
        """`force=true` hacia `DELETE FROM tracks WHERE filename = ?` sin filtro
        de propietario."""
        nombre = "track01.mp3"
        fp_ajeno = uuid.uuid4().hex
        _guardar(app_mod.db, filename=nombre, fingerprint=fp_ajeno,
                 artist="DJ ALPHA")

        # `force` es un QUERY param en /analyze, no Form. Mandarlo en `data=`
        # lo hace desaparecer sin error: FastAPI lo ignora, `force` queda False
        # y la rama del borrado nunca corre. Este test paso en verde asi hasta
        # que la mutacion (volver a borrar por filename) NO lo tumbo y delato
        # que no probaba nada.
        r = client.post("/analyze",
                        params={"force": "true"},
                        files={"file": (nombre, self._wav_bytes(freq=550.0),
                                        "audio/wav")})
        assert r.status_code == 200, r.text

        assert app_mod.db.get_track_by_fingerprint(fp_ajeno) is not None, (
            "SEC-01 VIVO: un force=true de otro usuario borro el track de A "
            "solo por compartir nombre de fichero"
        )


class TestFilasLegacySinHuella:
    def test_sin_huella_no_se_puede_verificar(self, db):
        """Una fila sin `fingerprint` no es verificable. La decision tomada es
        NO servirla como cache-hit: preferimos reanalizar a arriesgarnos a
        devolver el analisis de un desconocido."""
        _guardar(db, filename='viejo.mp3', fingerprint='',
                 artist='DESCONOCIDO', id_='legacy-001')
        fila = db.get_track_by_filename('viejo.mp3')
        guardada = (db._row_to_dict(fila) or {}).get('fingerprint')
        assert guardada in ('', None)
        assert guardada != uuid.uuid4().hex, (
            "una huella vacia nunca debe considerarse coincidencia"
        )
