"""Las dos medidas que corren en el ordenador del DJ, sobre su disco (estudio
de la competencia, 2026-11-05): la TONALIDAD de /analyze frente a la de sus
programas (`scripts/medir_tonalidad.py`) y el DESFASE de 26/50 ms de algunos
MP3 entre programas (`scripts/medir_desfase_mp3.py`). Las dos leen las
bibliotecas con `scripts/bibliotecas_dj.py`, que lee lo mismo y en las mismas
unidades que el import de la app.

    pytest test_las_medidas_en_el_disco_del_dj.py -v
"""

import os
import struct
import sys

import numpy as np
import pytest

import tonalidad as ton

# Se importan como los importa el propio script (desde `scripts/`), para que
# el proceso que analiza reciba la MISMA función: con otro módulo cargado a
# mano, el pickle de `ProcessPoolExecutor` no la reconoce.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

import bibliotecas_dj as bib  # noqa: E402
import medir_desfase_mp3 as md  # noqa: E402
import medir_tonalidad as mt  # noqa: E402


# --------------------------------------------------------------- las tonalidades

TEXTOS = ['Am', 'A#m', 'Bbm', 'B♭m', 'F# min', 'F#m', 'Gb', 'C major', 'C', 'Ebm',
          'E minor', '8A', '08A', '12B', '1m', '1d', '6m', 'G#', 'Abm', 'Dbm', 'db',
          'a#m', 'Cbm', 'E#', '13A', 'H', '', '?', 'Unknown', 'xyz']


def test_la_tonalidad_en_texto_dice_lo_mismo_que_el_servidor():
    """`camelot_de_texto` existe para no cargar el servidor en el disco del DJ;
    tiene que decir exactamente lo que `main.normalize_musical_key`."""
    import main
    for raw in TEXTOS:
        servidor = main.normalize_musical_key(raw)
        assert bib.camelot_de_texto(raw) == (servidor[1] if servidor else None), raw


def test_la_tabla_de_traktor():
    """0-11 mayores de Do a Si, 12-23 menores en el mismo orden (medido)."""
    assert bib.camelot_de_traktor('0') == '8B'     # Do mayor
    assert bib.camelot_de_traktor(9) == '11B'      # La mayor
    assert bib.camelot_de_traktor(21) == '8A'      # La menor
    assert bib.camelot_de_traktor(19) == '6A'      # Sol menor (Grasshopper)
    assert bib.camelot_de_traktor(15) == '2A'      # Mi bemol menor
    assert bib.camelot_de_traktor(24) is None
    assert bib.camelot_de_traktor('x') is None


# --------------------------------------------------------------- las bibliotecas

def test_rekordbox(tmp_path):
    xml = tmp_path / 'rekordbox.xml'
    xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0"><COLLECTION Entries="2">
 <TRACK TrackID="1" Name="Uno" Location="file://localhost/Users/dj/M%C3%BAsica/Uno%20Dos.mp3"
        AverageBpm="128.00" Tonality="Abm">
  <TEMPO Inizio="0.025" Bpm="128.00" Metro="4/4" Battito="1"/>
  <POSITION_MARK Name="" Type="0" Start="31.250" Num="0"/>
  <POSITION_MARK Name="" Type="0" Start="45.0" Num="-1"/>
  <POSITION_MARK Name="" Type="4" Start="60.0" End="61.875" Num="3"/>
 </TRACK>
 <TRACK TrackID="2" Name="Dos" Location="file://localhost/C:/Musica/dos.mp3" Tonality="8A"/>
</COLLECTION><PLAYLISTS><NODE Type="0" Name="ROOT"><NODE Name="L" Type="1">
 <TRACK Key="1"/></NODE></NODE></PLAYLISTS></DJ_PLAYLISTS>""", encoding='utf-8')
    uno, dos = bib.leer_rekordbox(str(xml))
    assert uno.ruta == '/Users/dj/Música/Uno Dos.mp3'
    assert uno.tonalidad == '1A' and uno.bpm == 128.0 and uno.ancla == 0.025
    assert [(c.segundos, c.tipo, c.pad) for c in uno.cues] == [
        (31.25, 'cue', 0), (45.0, 'cue', -1), (60.0, 'loop', 3)]
    assert dos.ruta == 'C:/Musica/dos.mp3' and dos.tonalidad == '8A'


def test_traktor(tmp_path):
    nml = tmp_path / 'collection.nml'
    nml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<NML VERSION="20"><COLLECTION ENTRIES="1">
 <ENTRY TITLE="Uno" ARTIST="X">
  <LOCATION DIR="/:-=MUSICA=-/:-=MINIMAL=-/:" FILE="uno.mp3" VOLUME="Untitled"/>
  <INFO KEY="Gm"/>
  <TEMPO BPM="128.000000"/>
  <MUSICAL_KEY VALUE="21"/>
  <CUE_V2 NAME="AutoGrid" DISPL_ORDER="0" TYPE="4" START="25.000000" LEN="0" HOTCUE="-1">
   <GRID BPM="128.000000"/></CUE_V2>
  <CUE_V2 NAME="n.n." TYPE="0" START="31250.0" LEN="0" HOTCUE="0"/>
  <CUE_V2 NAME="fade" TYPE="1" START="100.0" LEN="0" HOTCUE="-1"/>
  <CUE_V2 NAME="loop" TYPE="5" START="60000.0" LEN="1875.0" HOTCUE="2"/>
 </ENTRY>
</COLLECTION><PLAYLISTS/></NML>""", encoding='utf-8')
    (t,) = bib.leer_traktor(str(nml), existe=lambda r: r == '/Volumes/Untitled')
    assert t.ruta == '/Volumes/Untitled/-=MUSICA=-/-=MINIMAL=-/uno.mp3'
    # MUSICAL_KEY manda sobre el texto de INFO, como en la app.
    assert t.tonalidad == '8A'
    # El marcador de la rejilla va en MILISEGUNDOS.
    assert t.ancla == pytest.approx(0.025) and t.bpm == 128.0
    assert [(c.segundos, c.tipo, c.pad) for c in t.cues] == [
        (31.25, 'cue', 0), (60.0, 'loop', 2)]


def test_la_ruta_de_traktor_en_el_disco_de_arranque_y_en_windows():
    nada = lambda r: False
    assert bib.ruta_de_traktor('Macintosh HD', '/:Users/:dj/:', 'a.mp3', existe=nada) == \
        '/Users/dj/a.mp3'
    assert bib.ruta_de_traktor('Disco', '/:M/:', 'a.mp3', existe=nada) == '/Volumes/Disco/M/a.mp3'
    assert bib.ruta_de_traktor('C:', '/:Musica/:', 'a.mp3', existe=nada) == 'C:/Musica/a.mp3'


def test_virtualdj(tmp_path):
    xml = tmp_path / 'database.xml'
    # Con un byte que no es UTF-8 en el nombre de un cue, como en las bases de
    # datos de verdad: no puede tumbar la lectura.
    xml.write_bytes(b"""<?xml version="1.0" encoding="UTF-8"?>
<VirtualDJ_Database Version="2026">
 <Song FilePath="/Users/dj/uno.mp3" FileSize="1000">
  <Tags Author="X" Title="Uno" Key="Gm" />
  <Scan Version="801" Bpm="0.468750" Phase="1.896" Key="Am" />
  <Poi Name="Drop \xff" Pos="31.25" Num="1" Type="cue" />
  <Poi Pos="10.0" Type="automix" Point="realStart" />
  <Poi Name="Break 1" Pos="40.0" Type="remix" />
  <Poi Pos="60.0" Num="3" Type="loop" Size="8.0" />
 </Song>
 <Song FilePath="netsearch://dz123"><Scan Bpm="0.5" Key="C"/></Song>
 <Song FilePath="/Users/dj/dos.mp3"><Poi Pos="0.120" Type="beatgrid" /></Song>
</VirtualDJ_Database>""")
    uno, dos = bib.leer_virtualdj(str(xml))
    assert uno.ruta == '/Users/dj/uno.mp3'
    assert uno.tonalidad == '8A'                    # la del análisis (<Scan>)
    assert uno.bpm == pytest.approx(128.0) and uno.ancla == pytest.approx(1.896)
    assert [(c.segundos, c.tipo, c.pad) for c in uno.cues] == [
        (31.25, 'cue', 0), (60.0, 'loop', 2)]       # sin automix ni remix
    assert dos.ancla == pytest.approx(0.12)          # el POI de la rejilla, si no hay Phase


def test_el_mismo_fichero_casa_entre_programas_con_acentos_de_mac():
    import unicodedata
    compuesta = unicodedata.normalize('NFC', '/Users/dj/Música/Canción.mp3')
    descompuesta = unicodedata.normalize('NFD', compuesta)
    assert compuesta != descompuesta
    assert bib.clave_de_ruta(compuesta) == bib.clave_de_ruta(descompuesta.upper().lower())


def test_la_fase_es_modulo_el_beat():
    beat = 60 / 128.0
    assert bib.diferencia_de_fase(10.026, 10.0, beat) == pytest.approx(0.026)
    assert bib.diferencia_de_fase(10.0 + beat, 10.026, beat) == pytest.approx(-0.026)
    assert bib.diferencia_de_fase(0.0, beat - 0.01, beat) == pytest.approx(0.01)
    assert bib.diferencia_de_fase(5 * beat, 0.0, beat) == pytest.approx(0.0)


def test_los_cues_se_emparejan_uno_a_uno():
    a = [bib.Cue(10.0, 'cue'), bib.Cue(20.026, 'cue'), bib.Cue(50.0, 'cue')]
    b = [bib.Cue(20.0, 'cue'), bib.Cue(10.026, 'cue'), bib.Cue(80.0, 'cue')]
    pares = bib.emparejar_cues(a, b)
    assert [round(d, 3) for _, _, d in pares] == [-0.026, 0.026]


# --------------------------------------------------------------- la cabecera del MP3

def _trama_mp3(cuerpo=b''):
    """Una trama MPEG-1 capa III a 128 kbps y 44,1 kHz, estéreo: 417 bytes."""
    t = bytes([0xFF, 0xFB, 0x90, 0x00]) + cuerpo
    return t + bytes(417 - len(t))


def _mp3(tmp_path, nombre, primera, id3=True):
    datos = b''
    if id3:
        tam = 300
        datos += b'ID3\x03\x00\x00' + bytes([(tam >> 21) & 0x7F, (tam >> 14) & 0x7F,
                                              (tam >> 7) & 0x7F, tam & 0x7F]) + bytes(tam)
    datos += primera + _trama_mp3() * 3
    ruta = tmp_path / nombre
    ruta.write_bytes(datos)
    return str(ruta)


def _info(marca=b'Info', lame=True, retardo=576, relleno=1000):
    cuerpo = bytes(32) + marca + struct.pack('>I', 0x0F) + bytes(4 + 4 + 100 + 4)
    if lame:
        cuerpo += b'LAME3.100' + bytes(12) + bytes([retardo >> 4,
                                                    ((retardo & 0xF) << 4) | (relleno >> 8),
                                                    relleno & 0xFF])
    return _trama_mp3(cuerpo)


def test_la_cabecera_del_mp3(tmp_path):
    c = md.cabecera_mp3(_mp3(tmp_path, 'a.mp3', _info()))
    assert c['tipo'] == 'Info+LAME' and c['retardo'] == 576 and c['relleno'] == 1000
    assert c['id3'] == 310 and c['basura'] == 0 and c['codificador'] == 'LAME3.100'
    assert md.cabecera_mp3(_mp3(tmp_path, 'b.mp3', _info(b'Xing', lame=False)))['tipo'] == 'Xing'
    assert md.cabecera_mp3(_mp3(tmp_path, 'c.mp3', _trama_mp3(), id3=False))['tipo'] == \
        'sin cabecera'
    # Basura entre la etiqueta y la primera trama: se salta y se cuenta.
    c = md.cabecera_mp3(_mp3(tmp_path, 'd.mp3', b'\x00' * 7 + _info()))
    assert c['tipo'] == 'Info+LAME' and c['basura'] == 7
    assert md.cabecera_mp3(str(tmp_path / 'e.flac'))['tipo'] == 'no es MP3'
    assert md.cabecera_mp3(str(tmp_path / 'no_existe.mp3'))['tipo'] == 'sin fichero'


def test_el_desfase_se_reparte_por_cabecera():
    def tema(programa, ancla, cues):
        return bib.TemaDeUnPrograma(programa, '/x', bpm=128.0, ancla=ancla,
                                    cues=[bib.Cue(s, 'cue') for s in cues])
    por_fichero = {
        'bueno': {'rekordbox': tema('rekordbox', 0.100, [10.0, 20.0]),
                  'traktor': tema('traktor', 0.101, [10.0, 20.0])},
        'raro': {'rekordbox': tema('rekordbox', 0.126, [10.026, 20.026]),
                 'traktor': tema('traktor', 0.100, [10.0, 20.0])},
    }
    cab = {'bueno': {'tipo': 'Info+LAME'}, 'raro': {'tipo': 'Xing'}}
    par = md.comparar(por_fichero, cab)['rekordbox ~ traktor']
    r = md.resumen(par['anclas'])
    assert r['Xing']['cajones']['≈26'] == 1 and r['Info+LAME']['cajones']['≈0'] == 1
    r = md.resumen(par['cues'])
    assert r['Xing']['mediana'] == pytest.approx(26.0)
    assert r['TODOS']['n'] == 4
    # Un BPM que no casa (ni doble) no se compara: la fase no diría nada.
    a, b = tema('rekordbox', 0.1, []), tema('traktor', 0.1, [])
    b.bpm = 130.0
    assert md.beat_comun(a, b) is None
    b.bpm = 64.0
    assert md.beat_comun(a, b) == pytest.approx(60 / 128.0)


# --------------------------------------------------------------- la tonalidad

SR = 22050


def _acorde(segundos, notas_hz=(110.0, 220.0, 261.63, 329.63), sr=SR):
    """La menor: La grave y la tríada (La, Do, Mi) con armónicos."""
    t = np.arange(int(segundos * sr)) / sr
    y = sum(sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3)) for f in notas_hz)
    y += np.random.RandomState(1).randn(len(t)) * 0.01
    return (y / np.max(np.abs(y)) * 0.5).astype(np.float32)


def test_el_croma_todo_es_lo_que_hace_hoy_analyze():
    import librosa
    y = _acorde(12)
    c = mt.cromas_del_audio(y, SR)
    hoy = librosa.feature.chroma_cqt(y=y, sr=SR, n_chroma=12, n_octaves=7).mean(axis=1)
    assert np.allclose(c['todo'], hoy / hoy.sum(), atol=1e-6)
    assert set(mt.CROMAS) <= set(c)
    assert c['trozos']                      # al menos un minuto… o lo que haya


def test_un_acorde_de_la_menor_sale_la_menor():
    c = mt.cromas_del_audio(_acorde(12), SR)
    for nombre in mt.CROMAS:
        k, r = mt.tonalidad_y_correlacion(c[nombre], ton.PERFILES['kk'])
        assert k == '8A', nombre
    assert mt.por_voto(c['trozos'], ton.PERFILES['kk']) == '8A'


def test_el_consenso_y_lo_que_ve_el_dj():
    assert mt.consenso({'rekordbox': '8A', 'etiqueta': '8A', 'traktor': '9A'}) == '8A'
    assert mt.consenso({'rekordbox': '8A', 'etiqueta': '9A'}) is None
    assert mt.consenso({'rekordbox': '8A', 'traktor': '8A', 'etiqueta': '9A',
                        'virtualdj': '9A'}) is None
    r = mt.referencias({'traktor': '6A', 'etiqueta': '6A'})
    assert r['programa'] == '6A' and r['consenso'] == '6A' and r['rekordbox'] is None


def test_el_aprendido_no_se_mide_con_lo_que_ha_visto():
    # Hacen falta temas en mayor y en menor: sin uno de los dos modos no hay
    # perfil que aprender (`tonalidad.aprender_perfiles`).
    rnd = np.random.RandomState(3)
    temas, verdad = [], []
    for i in range(80):
        tonica, modo = rnd.randint(12), ('mayor', 'menor')[i % 2]
        perfil = np.asarray(ton.PERFILES['edma'][0 if modo == 'mayor' else 1])
        croma = np.roll(perfil + rnd.rand(12) * 0.2, tonica)
        temas.append({'ruta': f'/m/{i}.mp3', 'cromas': {n: list(croma / croma.sum())
                                                       for n in mt.CROMAS}})
        verdad.append(ton.camelot_de(tonica, modo))
    pred = mt.aprendidas(temas, verdad)['todo + aprendido']
    d = mt.contar(pred, verdad)
    assert d['comparados'] == 80 and d['igual'] >= 70


def test_la_tabla_siempre_trae_lo_de_hoy():
    verdad = ['8A'] * 10
    pred = {f'v{i}': ['8A'] * 10 for i in range(20)}
    pred[mt.LO_DE_HOY] = ['9A'] * 10
    filas = mt.tabla(pred, verdad, mejores=5)
    assert len(filas) == 6 and filas[-1][0] == mt.LO_DE_HOY


def test_los_perfiles_de_la_electronica_estan_en_la_sombra():
    for nombre in ('shaath', 'edma', 'edmm', 'braw', 'bgate'):
        assert nombre in ton.PERFILES
    for nombre, (mayor, menor) in ton.PERFILES.items():
        if nombre == 'diatonico':
            continue    # La menor natural son las notas de Do mayor: empatan
        for tonica in (0, 7):
            assert ton.tonalidad_desde_croma(np.roll(menor, tonica), mayor, menor) == \
                ton.camelot_de(tonica, 'menor'), nombre
            if len(set(mayor)) > 1:          # el mayor de edmm es plano a propósito
                assert ton.tonalidad_desde_croma(np.roll(mayor, tonica), mayor, menor) == \
                    ton.camelot_de(tonica, 'mayor'), nombre
    r = ton.evaluar_perfiles([('ab12', np.roll(ton.PERFILES['kk'][1], 9), '8A')])
    assert set(ton.PERFILES) | {'aprendido'} == set(r['perfiles'])


def test_la_medida_de_punta_a_punta(tmp_path, capsys):
    """Un Rekordbox con un tema que está y otro que no: mide el que está, no se
    cae con el otro y deja el CSV."""
    import soundfile as sf
    wav = tmp_path / 'la menor.wav'
    sf.write(str(wav), _acorde(35), SR)
    xml = tmp_path / 'rekordbox.xml'
    xml.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS><COLLECTION>
 <TRACK Location="file://localhost{str(wav).replace(' ', '%20')}" Tonality="Am"/>
 <TRACK Location="file://localhost/no/existe.mp3" Tonality="Cm"/>
</COLLECTION></DJ_PLAYLISTS>""", encoding='utf-8')
    salida_csv = tmp_path / 'detalle.csv'
    assert mt.main(['--rekordbox', str(xml), '--procesos', '1', '--csv', str(salida_csv)]) == 0
    texto = capsys.readouterr().out
    assert '1 temas con fichero' in texto
    assert '=== Tonalidad' in texto and mt.LO_DE_HOY in texto
    filas = salida_csv.read_text(encoding='utf-8').splitlines()
    assert len(filas) == 2 and 'la menor.wav' in filas[1]


# Una biblioteca que no está (el owner no tenía exportado su rekordbox.xml el
# 2026-11-06, y el script se cortaba con un FileNotFoundError) no corta la
# medida: se dice cómo conseguirla y se sigue con las demás.
_NML_DE_UNO = """<?xml version="1.0" encoding="UTF-8"?>
<NML VERSION="20"><COLLECTION ENTRIES="1">
 <ENTRY TITLE="Uno" ARTIST="X">
  <LOCATION DIR="/:Users/:dj/:" FILE="uno.mp3" VOLUME="Macintosh HD"/>
  <MUSICAL_KEY VALUE="21"/>
 </ENTRY>
</COLLECTION><PLAYLISTS/></NML>"""


def test_una_biblioteca_que_no_esta_no_corta_la_medida(tmp_path):
    nml = tmp_path / 'collection.nml'
    nml.write_text(_NML_DE_UNO, encoding='utf-8')
    roto = tmp_path / 'database.xml'
    roto.write_text('<VirtualDJ_Database><Song', encoding='utf-8')
    avisos, leidas = [], {}
    por_fichero = bib.leer_bibliotecas(
        rekordbox=str(tmp_path / 'rekordbox.xml'), traktor=str(nml),
        virtualdj=str(roto), leidas=leidas, avisar=avisos.append)
    assert leidas == {'traktor': 1}
    assert [list(v) for v in por_fichero.values()] == [['traktor']]
    assert len(avisos) == 2
    assert 'no está' in avisos[0] and 'Exportar colección en formato xml' in avisos[0]
    assert 'no se puede leer' in avisos[1]


def test_el_desfase_dice_que_hacen_falta_dos_bibliotecas(tmp_path, capsys):
    nml = tmp_path / 'collection.nml'
    nml.write_text(_NML_DE_UNO, encoding='utf-8')
    salida = md.main(['--rekordbox', str(tmp_path / 'rekordbox.xml'),
                      '--traktor', str(nml), '--csv', str(tmp_path / 'd.csv')])
    texto = capsys.readouterr().out
    assert salida == 1
    assert 'no está' in texto
    assert 'Hacen falta dos bibliotecas' in texto


# --------------------------------------------------------------- sobre el audio
#
# Lo que decide qué hay que corregir no es lo que se separan dos programas,
# sino dónde cae la rejilla de cada uno sobre el audio tal como lo leen los
# decodificadores de DJ Analyzer (2026-11-06: Rekordbox y VirtualDJ se
# separaban +45-52 ms en los MP3 con cabecera LAME del owner, y 0 en los
# demás).

def _ffmpeg_con_lame():
    import shutil
    import subprocess
    f = shutil.which('ffmpeg')
    if not f:
        return None
    r = subprocess.run([f, '-hide_banner', '-encoders'], capture_output=True, text=True)
    return f if 'libmp3lame' in r.stdout else None


def _tema_con_bombo(segundos=40.0, sr=44100, bpm=128.0, t0=0.5, hats=True):
    """Un bombo (un barrido de 130 a 50 Hz que se apaga) en cada beat desde `t0`,
    y un charles en cada contratiempo."""
    rng = np.random.RandomState(3)
    n = int(segundos * sr)
    y = np.zeros(n)
    k = np.arange(int(0.12 * sr)) / sr
    bombo = np.sin(2 * np.pi * (50 + 80 * np.exp(-k * 40)) * k) * np.exp(-k * 18)
    hat = rng.randn(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / sr * 120) * 0.15
    beat, j = 60.0 / bpm, 0
    while t0 + (j + 1) * beat < segundos - 0.2:
        a = int(round((t0 + j * beat) * sr))
        y[a:a + len(bombo)] += 0.8 * bombo
        if hats:
            h = int(round((t0 + (j + 0.5) * beat) * sr))
            y[h:h + len(hat)] += hat
        j += 1
    y += rng.randn(n) * 0.003
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def _mp3s(tmp_path, ffmpeg):
    """El mismo tema en un MP3 con trama Info y etiqueta LAME (CBR) y en uno
    sin trama (`-write_xing 0`): las mismas tramas de audio, sin la cabecera."""
    import subprocess
    import soundfile as sf
    wav = tmp_path / 'tema.wav'
    sf.write(str(wav), _tema_con_bombo(), 44100, subtype='PCM_16')
    salida = {}
    for nombre, extra in (('info', []), ('sin', ['-write_xing', '0'])):
        ruta = tmp_path / f'{nombre}.mp3'
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
                        '-c:a', 'libmp3lame', '-b:a', '320k', *extra, str(ruta)], check=True)
        salida[nombre] = str(ruta)
    return salida


def test_el_peine_no_salta_al_mover_el_audio():
    """Mover el audio unas muestras mueve la medida lo mismo, no 4 ms: con el
    valor absoluto de los graves en vez de la envolvente analítica, el bombo
    salía con un rizo y el peine saltaba de un rizo a otro."""
    sr, t0 = 44100, 0.5
    y = _tema_con_bombo(segundos=30)
    medidas = []
    for s in (0, 3, 7, 11, 16, 25, 31):
        corrido = np.concatenate([np.zeros(s, np.float32), y])
        r = md.desfase_sobre_el_audio(md.golpes(corrido, sr), md.PASO / sr, 128.0, t0 + s / sr)
        assert r is not None and r[1] > md.NITIDEZ_MINIMA
        medidas.append(r[0] * 1000)
    assert max(medidas) - min(medidas) < 1.0, medidas
    # Una rejilla puesta 20 ms tarde se mide 20 ms tarde, y medio beat es el borde.
    r = md.desfase_sobre_el_audio(md.golpes(y, sr), md.PASO / sr, 128.0, t0 + 0.020)
    assert r[0] * 1000 == pytest.approx(20.0 + medidas[0], abs=1.0)


def test_ffmpeg_se_salta_la_cabecera_lame_y_un_programa_ingenuo_no(tmp_path):
    """Medido con un clic (2026-11-06): ffmpeg, el de DJ Analyzer en el
    escritorio, se salta la trama Info y el retardo de la etiqueta LAME (el
    tema sale en su muestra) y sin cabecera sale 25,1 ms tarde (1.105 muestras:
    no hay de dónde saber el retardo). Una rejilla de un programa que no se
    salta nada cae 51,2 ms tarde en el de cabecera LAME y en su sitio en el
    otro; la de uno que se lo salta, en su sitio en los dos."""
    ffmpeg = _ffmpeg_con_lame()
    if not ffmpeg:
        pytest.skip('sin ffmpeg con libmp3lame')
    sr, t0 = 44100, 0.5
    f = _mp3s(tmp_path, ffmpeg)
    assert md.cabecera_mp3(f['info'])['tipo'] == 'Info+LAME'
    assert md.cabecera_mp3(f['sin'])['tipo'] == 'sin cabecera'
    decs = {'ffmpeg': ffmpeg}
    medidas = {}
    for nombre, ruta in f.items():
        con_lame = nombre == 'info'
        ingenuo = t0 + (2257 if con_lame else 1105) / sr
        el_de_ffmpeg = t0 + (0 if con_lame else 1105) / sr
        r = md.medir_un_fichero(ruta, sr, {'rekordbox': (128.0, ingenuo),
                                           'virtualdj': (128.0, el_de_ffmpeg)}, decs)
        medidas[nombre] = {p: v[0] * 1000 for p, v in r['desfases']['ffmpeg'].items()}
    assert medidas['info']['rekordbox'] - medidas['sin']['rekordbox'] == \
        pytest.approx(51.2, abs=2.0)
    assert medidas['info']['virtualdj'] - medidas['sin']['virtualdj'] == pytest.approx(0, abs=2.0)
    # mpg123 (soundfile), el del análisis del servidor, lee lo mismo que ffmpeg.
    if 'soundfile' in md.decodificadores_disponibles(ffmpeg):
        r = md.medir_un_fichero(f['info'], sr, {}, {'ffmpeg': ffmpeg, 'soundfile': 'soundfile'})
        assert r['retrasos'].get('soundfile') == pytest.approx(0.0, abs=1e-4)


def test_la_medida_sobre_el_audio_de_punta_a_punta(tmp_path, capsys):
    """Con un solo programa y `--audio`: mide, saca la tabla por cabecera y
    deja el CSV."""
    ffmpeg = _ffmpeg_con_lame()
    if not ffmpeg:
        pytest.skip('sin ffmpeg con libmp3lame')
    f = _mp3s(tmp_path, ffmpeg)
    pistas = ''.join(
        f' <TRACK Location="file://localhost{ruta}" AverageBpm="128.00">'
        f'<TEMPO Inizio="{0.5 + (2257 if n == "info" else 1105) / 44100:.3f}" Bpm="128.00" '
        f'Metro="4/4" Battito="1"/></TRACK>\n'
        for n, ruta in f.items())
    xml = tmp_path / 'rekordbox.xml'
    xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<DJ_PLAYLISTS><COLLECTION>\n'
                   + pistas + '</COLLECTION></DJ_PLAYLISTS>', encoding='utf-8')
    detalle = tmp_path / 'audio.csv'
    salida = md.main(['--rekordbox', str(xml), '--audio', '5', '--procesos', '1',
                      '--ffmpeg', ffmpeg, '--csv-audio', str(detalle)])
    texto = capsys.readouterr().out
    assert salida == 0, texto
    assert 'rekordbox sobre el audio de ffmpeg: 2 temas' in texto
    linea = next(x for x in texto.splitlines() if x.strip().startswith('Info+LAME'))
    assert '+51' in linea or '+50' in linea or '+52' in linea, linea
    filas = detalle.read_text(encoding='utf-8').splitlines()
    assert filas[0].startswith('ruta,cabecera') and len(filas) >= 3


# --------------------------------------------------------------- el resumen del CSV

def test_el_resumen_del_csv(tmp_path, capsys):
    """Sin volver a analizar: separa mayor de menor (un perfil que lo dice todo
    en menor acierta los menores y ninguno de los mayores) y arma lo de hoy en
    Render (el voto de los trozos pasados 4 min)."""
    cols = ['ruta', 'duracion', 'rekordbox', 'traktor', 'virtualdj', 'etiqueta', 'consenso',
            'log + edmm', 'todo + kk', 'voto + kk']
    filas = [
        # corto: hoy (todo + kk) acierta
        ['a.mp3', '200', '8A', '', '', '', '', '8A', '8A', '3A'],
        # largo: hoy en Render es el voto (falla), el tema entero acierta
        ['b.mp3', '400', '8A', '', '', '', '', '8A', '8A', '9A'],
        # mayor: el perfil que tira a menor da la relativa
        ['c.mp3', '380', '8B', '', '', '', '', '8A', '8B', '8B'],
        ['d.mp3', '390', '', '', '', '', '', '5A', '5A', '5A'],
    ]
    ruta = tmp_path / 'medir.csv'
    import csv as _csv
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(cols)
        w.writerows(filas)
    fila = dict(zip(cols, filas[1]))
    assert mt.lo_de_hoy_en_render(fila, 400) == '9A'
    assert mt.lo_de_hoy_en_render(dict(zip(cols, filas[0])), 200) == '8A'

    assert mt.main(['--resumen', str(ruta)]) == 0
    texto = capsys.readouterr().out
    assert 'rekordbox 3' in texto and '67 % en menor' in texto
    hoy = next(x for x in texto.splitlines() if x.strip().startswith('hoy en Render'))
    assert hoy.split()[3] == '67', hoy        # a y c bien, b mal
    edmm = next(x for x in texto.splitlines() if x.strip().startswith('log + edmm'))
    # 67 % igual · 100 % mezcla · 100 % en menor · 0 % en mayor · dice menor 100 %
    assert edmm.split()[3:13:2] == ['67', '100', '100', '0', '100'], edmm
    assert 'Más de 4 min (Render: el voto de los trozos): 2 temas' in texto


# --------------------------------------------------------------- la pista de prueba
#
# Con música de verdad no se puede separar el sesgo del detector de lo que
# cuenta de más cada programa (2026-11-06: ¿51 o 39 ms para Rekordbox?). La
# pista de prueba es el mismo bombo sintético en un WAV y en cuatro MP3 con
# distinta cabecera: lo que un programa ponga en el WAV es su sesgo.

import pista_de_prueba as pp  # noqa: E402


def test_la_pista_resta_el_sesgo_de_cada_programa():
    anclas = {'wav': 0.603, 'sin_cabecera': 0.628, 'info_lame': 0.654,
              'xing_lame': 0.654 + pp.BEAT, 'xing': None}
    d = pp.donde_cae(anclas)
    assert d['wav'] == 0.0
    assert d['sin_cabecera'] == pytest.approx(25.0, abs=0.01)
    assert d['info_lame'] == pytest.approx(51.0, abs=0.01)
    assert d['xing_lame'] == pytest.approx(51.0, abs=0.01), 'un beat más es la misma rejilla'
    assert d['xing'] is None
    assert pp.donde_cae({'info_lame': 0.654})['info_lame'] is None, 'sin el WAV no hay sesgo que restar'


def test_la_pista_lleva_las_cuatro_cabeceras_y_ffmpeg_las_lee_como_en_tu_disco(tmp_path):
    """La etiqueta que deja ffmpeg dice «Lavc…»; se cambia por «LAME3.100» con su
    CRC rehecho, y los decodificadores la siguen leyendo (si el CRC estuviera mal
    la ignorarían y el golpe saldría 25 ms tarde). Y la de Xing sin etiqueta sale
    como en el disco del owner: 25,1 ms tarde con ffmpeg y 13,1 con mpg123."""
    ffmpeg = _ffmpeg_con_lame()
    if not ffmpeg:
        pytest.skip('sin ffmpeg con libmp3lame')
    assert pp.crear(str(tmp_path), ffmpeg) == 0
    for v, tipo in pp.VARIANTES.items():
        if tipo:
            assert md.cabecera_mp3(str(tmp_path / pp.nombre(v)))['tipo'] == tipo, v
    for v in ('info_lame', 'xing_lame'):
        c = md.cabecera_mp3(str(tmp_path / pp.nombre(v)))
        assert c['codificador'] == 'LAME3.100'
        pos_crc = c['pos_codificador'] - c['trama'] + 34
        with open(tmp_path / pp.nombre(v), 'rb') as f:
            f.seek(c['trama'])
            trama = f.read(192)
        assert int.from_bytes(trama[pos_crc:pos_crc + 2], 'big') == \
            pp.crc_de_la_etiqueta(trama, pos_crc), v

    app = pp.donde_cae_en_la_app(str(tmp_path), ffmpeg)
    assert app['ffmpeg']['info_lame'] == pytest.approx(0.0, abs=0.05)
    assert app['ffmpeg']['xing_lame'] == pytest.approx(0.0, abs=0.05)
    assert app['ffmpeg']['sin_cabecera'] == pytest.approx(25.06, abs=0.05)
    assert app['ffmpeg']['xing'] == pytest.approx(25.06, abs=0.05)
    if 'soundfile' in app:
        assert app['soundfile']['info_lame'] == pytest.approx(0.0, abs=0.05)
        assert app['soundfile']['xing'] == pytest.approx(13.06, abs=0.05)


def test_el_crc_de_la_etiqueta_es_el_de_ffmpeg(tmp_path):
    """El CRC que rehace la pista es el mismo que calcula ffmpeg al escribir la
    etiqueta: se comprueba sobre la que deja él, antes de tocarla, en un MP3
    mono (el CRC cae DENTRO de los 190 bytes y cuenta como ceros) y en uno
    estéreo (cae justo fuera). Solo el CBR: en el VBR ffmpeg reescribe la trama
    al terminar y su CRC no casa con ella."""
    import subprocess
    import soundfile as sf
    ffmpeg = _ffmpeg_con_lame()
    if not ffmpeg:
        pytest.skip('sin ffmpeg con libmp3lame')
    y = _tema_con_bombo(segundos=5)
    for canales, senal in ((1, y), (2, np.stack([y, y], axis=1))):
        wav = tmp_path / f'a{canales}.wav'
        sf.write(str(wav), senal, 44100, subtype='PCM_16')
        mp3 = tmp_path / f'a{canales}.mp3'
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
                        '-c:a', 'libmp3lame', '-b:a', '320k', str(mp3)], check=True)
        c = md.cabecera_mp3(str(mp3))
        pos_crc = c['pos_codificador'] - c['trama'] + 34
        assert pos_crc == (175 if canales == 1 else 190)
        with open(mp3, 'rb') as f:
            f.seek(c['trama'])
            trama = f.read(192)
        assert int.from_bytes(trama[pos_crc:pos_crc + 2], 'big') == \
            pp.crc_de_la_etiqueta(trama, pos_crc), canales


def test_la_lectura_de_la_pista_de_punta_a_punta(tmp_path, capsys):
    ffmpeg = _ffmpeg_con_lame()
    if not ffmpeg:
        pytest.skip('sin ffmpeg con libmp3lame')
    assert pp.crear(str(tmp_path), ffmpeg) == 0
    inizio = {'wav': 0.603, 'sin_cabecera': 0.628, 'info_lame': 0.654,
              'xing_lame': 0.654, 'xing': 0.654}
    pistas = ''.join(
        f' <TRACK Location="file://localhost{tmp_path / pp.nombre(v)}" AverageBpm="125.00">'
        f'<TEMPO Inizio="{a:.3f}" Bpm="125.00" Metro="4/4" Battito="1"/></TRACK>\n'
        for v, a in inizio.items())
    xml = tmp_path / 'rekordbox.xml'
    xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<DJ_PLAYLISTS><COLLECTION>\n'
                   + pistas + '</COLLECTION></DJ_PLAYLISTS>', encoding='utf-8')
    assert pp.main(['--leer', str(tmp_path), '--rekordbox', str(xml), '--ffmpeg', ffmpeg]) == 0
    texto = capsys.readouterr().out
    assert '=== Pista de prueba' in texto
    fila = next(x for x in texto.splitlines() if x.strip().startswith('Info+LAME'))
    assert '+51.0' in fila, fila
    assert 'Lo que hay que restar' in texto


# La segunda vuelta de la pista (2026-11-10): el Rekordbox de hoy puso los
# cuatro MP3 en el mismo sitio (no cuenta la trama Info), y las rejillas que el
# owner ya tenía dicen que sí la cuenta. Se resuelve con copias de sus propios
# MP3: el programa las analiza hoy como temas nuevos, y lo que su rejilla se
# mueve respecto a la que tenía es lo que ha cambiado.

def _rekordbox_con(tmp_path, nombre_xml, pistas, version='6.8.5'):
    """Un rekordbox.xml con [(ruta, inizio, bpm)]."""
    cuerpo = ''.join(
        f' <TRACK Location="file://localhost{str(ruta).replace(" ", "%20")}" AverageBpm="{bpm:.2f}">'
        f'<TEMPO Inizio="{inizio:.3f}" Bpm="{bpm:.2f}" Metro="4/4" Battito="1"/></TRACK>\n'
        for ruta, inizio, bpm in pistas)
    xml = tmp_path / nombre_xml
    xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<DJ_PLAYLISTS Version="1.0.0">'
                   f'<PRODUCT Name="rekordbox" Version="{version}" Company="AlphaTheta"/>'
                   f'<COLLECTION>\n{cuerpo}</COLLECTION></DJ_PLAYLISTS>', encoding='utf-8')
    return str(xml)


def test_las_copias_de_la_biblioteca_dicen_lo_que_se_mueve_cada_programa(tmp_path, capsys):
    biblio, pista = tmp_path / 'biblio', tmp_path / 'pista'
    biblio.mkdir()
    pista.mkdir()
    con = _mp3(biblio, 'con lame.mp3', _info())
    sin = _mp3(biblio, 'viejo.mp3', _trama_mp3(), id3=False)
    de_la_pista = _mp3(pista, pp.nombre('info_lame'), _info())
    antes = _rekordbox_con(tmp_path, 'antes.xml', [(con, 0.551, 128.0), (sin, 0.525, 128.0),
                                                    (de_la_pista, 0.630, 125.0)])
    copias = pp.copiar_reales(str(pista), bib.leer_bibliotecas(rekordbox=antes), 5)
    assert sorted(c['cabecera'] for c in copias) == ['Info+LAME', 'sin cabecera'], \
        'una de cada cabecera, y los ficheros de la propia pista no son de la biblioteca'
    assert all((pista / c['copia']).exists() for c in copias)
    assert {c['cabecera']: c['programas']['rekordbox']['ancla'] for c in copias} == \
        {'Info+LAME': 0.551, 'sin cabecera': 0.525}
    # Otra llamada no elige otros: los programas ya tienen analizados esos.
    assert [c['copia'] for c in pp.copiar_reales(str(pista), {}, 5)] == [c['copia'] for c in copias]

    # Hoy el programa pone la del LAME 26 ms antes (al doble de BPM, que es la
    # misma rejilla) y la otra donde estaba.
    por_tipo = {c['cabecera']: str(pista / c['copia']) for c in copias}
    hoy = _rekordbox_con(tmp_path, 'hoy.xml', [(por_tipo['Info+LAME'], 0.525, 256.0),
                                                (por_tipo['sin cabecera'], 0.525, 128.0)],
                         version='7.0.4')
    movidos = pp.lo_que_se_mueven(copias, bib.leer_bibliotecas(rekordbox=hoy), str(pista))
    ms = {tipo: v for _, tipo, v in movidos['rekordbox']}
    assert ms['Info+LAME'] == pytest.approx(-26.0, abs=0.01)
    assert ms['sin cabecera'] == pytest.approx(0.0, abs=0.01)
    assert pp.version_de('rekordbox', hoy) == '7.0.4'
    assert pp.version_de('rekordbox', str(tmp_path / 'no_esta.xml')) is None

    assert pp.leer(str(pista), rekordbox=hoy) == 0
    texto = capsys.readouterr().out
    assert 'Versiones: rekordbox 7.0.4' in texto
    assert 'copias de tu biblioteca' in texto
    fila = next(x for x in texto.splitlines() if x.strip().startswith('Info+LAME') and 'ms' in x)
    assert '-26.0 ms' in fila, fila


def test_la_pista_no_reescribe_lo_que_ya_esta(tmp_path):
    """Volver a crearla (para añadir las copias) no toca los ficheros que los
    programas ya tienen analizados."""
    ffmpeg = _ffmpeg_con_lame()
    if not ffmpeg:
        pytest.skip('sin ffmpeg con libmp3lame')
    assert pp.crear(str(tmp_path), ffmpeg) == 0
    antes = {f: os.stat(tmp_path / f).st_mtime_ns for f in os.listdir(tmp_path)}
    assert pp.crear(str(tmp_path), ffmpeg) == 0
    assert {f: os.stat(tmp_path / f).st_mtime_ns for f in os.listdir(tmp_path)} == antes
