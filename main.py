"""
Transcriptor Whisper — v1.5.1

v1.5.1
  - Arreglo: en modo rápido los subtítulos salían como oraciones enormes.
    El modo por lotes, por defecto, no genera tiempos intermedios y cada bloque
    de hasta 30 s quedaba como un solo segmento. Ahora se piden los tiempos.
  - Subtítulos (SRT, VTT y video) armados a partir del tiempo de cada palabra:
    máximo 2 líneas de 42 caracteres y 6 segundos por subtítulo, cortando en
    los puntos y en las pausas. Los TXT no cambian.

v1.5
  - Archivos guardados: vista previa al marcar un archivo (texto o video).
    Los videos con subtítulos incrustados se ven con los subtítulos en el navegador.
  - Archivos guardados: filtro por tipo, "Abrir con el programa predeterminado" y
    "Mostrar en la carpeta"; la lista se actualiza al entrar a la pestaña
  - Si el archivo ya fue procesado, pregunta si sobrescribir o guardar como copia (_2, _3…)
  - Arreglo: con nombres que tienen puntos (ej. "ssstwitter.com_123.mp4") el TXT plano,
    SRT, VTT y JSON se guardaban con el nombre cortado ("ssstwitter.txt")

v1.4
  - Ruta en esta PC con "Examinar…", vista previa del video, vocabulario técnico,
    modo rápido por lotes, aviso de memoria insuficiente

v1.3
  - Copia del video con subtítulos (incrustados o grabados en la imagen) vía ffmpeg

v1.2.1
  - Arreglo: error "Value 's' is not in the list of choices" al eliminar archivos

v1.2
  - Descripción de cada modelo, pestaña "Archivos guardados" (ver, descargar, eliminar)

v1.1
  - Interfaz web con Gradio, selector de archivo, modelo, idioma y formatos
  - Barra de progreso, modelo cacheado entre transcripciones, carpeta "salidas"

v1.0
  - Script base por consola con faster-whisper y tqdm
"""

import io
import os
import sys
import json
import shutil
import inspect
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import nvidia.cublas, nvidia.cudnn

for pkg in (nvidia.cublas, nvidia.cudnn):
    bin_dir = Path(pkg.__path__[0]) / "bin"
    os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]

import gradio as gr
from PIL import Image
from faster_whisper import WhisperModel, BatchedInferencePipeline

VERSION = "1.5.1"

SALIDAS = Path(__file__).parent / "salidas"
SALIDAS.mkdir(exist_ok=True)
VISTA = SALIDAS / ".vista"   # subtítulos extraídos para el reproductor (no aparece en la lista)

MODELOS = {
    "large-v3": "**Máxima precisión.** El más lento. Recomendado para audio con ruido, "
                "varias voces o vocabulario técnico. Usa unos 5 GB de VRAM.",
    "large-v3-turbo": "**Casi la misma calidad que large-v3, varias veces más rápido.** "
                      "La mejor opción para el uso diario. Usa unos 3 GB de VRAM.",
    "medium": "**Intermedio.** Bien para audio limpio con una sola voz (clases, podcasts). "
              "Comete más errores con nombres propios. Usa unos 2 GB de VRAM.",
    "small": "**Rápido y liviano.** Para pruebas o si tenés poca VRAM. "
             "Se nota la pérdida de calidad en audios largos. Usa alrededor de 1 GB de VRAM.",
}
IDIOMAS = {"Español": "es", "Inglés": "en", "Detectar automáticamente": None}

# Formato -> sufijo del archivo generado
SUFIJOS = {
    "TXT con tiempos": "_tiempos.txt",
    "TXT plano": ".txt",
    "SRT": ".srt",
    "VTT": ".vtt",
    "JSON": ".json",
}
FORMATOS = list(SUFIJOS)

EXT_VIDEO = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".ts"}
EXT_MP4 = {".mp4", ".m4v", ".mov"}
EXT_NAVEGADOR = {".mp4", ".m4v", ".mov", ".webm"}   # los que el navegador puede reproducir
EXT_TEXTO = {".txt", ".srt", ".vtt", ".json", ".md"}
FILTROS = {
    "Todos": None,
    "Texto": {".txt", ".json", ".md"},
    "Subtítulos": {".srt", ".vtt"},
    "Videos": EXT_VIDEO,
}
LIMITE_TEXTO = 200_000   # caracteres que se muestran en la vista previa

MODOS_SUB = {
    "Incrustados (rápido, se activan y desactivan en el reproductor)": "soft",
    "Grabados en la imagen (lento, se ven en cualquier reproductor)": "hard",
}
LANG3 = {"es": "spa", "en": "eng"}

# Límites para armar subtítulos legibles
SUB_CARACTERES = 42    # por línea
SUB_LINEAS = 2
SUB_DURACION = 6.0     # segundos máximos en pantalla
SUB_PAUSA = 0.8        # una pausa más larga que esto corta el subtítulo
SUB_MINIMO = 1.0       # segundos mínimos en pantalla

# Diálogo nativo de Windows para elegir archivo. Corre en un proceso aparte
# porque tkinter no se lleva bien con los hilos de Gradio.
DIALOGO = r'''
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
p = filedialog.askopenfilename(
    title="Elegí un video o audio",
    filetypes=[
        ("Video y audio", "*.mp4 *.mkv *.mov *.m4v *.avi *.webm *.wmv *.flv *.ts "
                          "*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.wma"),
        ("Todos los archivos", "*.*"),
    ],
)
print(p, end="")
'''

_cache = {"nombre": None, "modelo": None}


# ---------- Utilidades ----------

def hms(t):
    """00:05:01"""
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"


def tiempo_sub(t, sep):
    """00:05:01,250 (SRT) o 00:05:01.250 (VTT)"""
    return f"{hms(t)}{sep}{int((t % 1) * 1000):03}"


def tamano(bytes_):
    for unidad in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.0f} {unidad}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def fecha(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y %H:%M")


# ---------- Entrada y vista previa del video ----------

def limpiar_ruta(ruta):
    # "Copiar como ruta" de Windows agrega comillas
    return (ruta or "").strip().strip('"').strip("'")


def resolver_entrada(archivo, ruta, estricto=True):
    """La ruta escrita tiene prioridad sobre el archivo subido."""
    ruta = limpiar_ruta(ruta)
    if ruta:
        p = Path(ruta)
        if p.is_file():
            return str(p)
        if estricto:
            raise gr.Error(f"No existe el archivo: {ruta}")
        return None
    if archivo:
        return archivo
    if estricto:
        raise gr.Error("Elegí un archivo: subilo, pegá la ruta o usá «Examinar…».")
    return None


def examinar(actual):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", DIALOGO], capture_output=True,
                       text=True, encoding="utf-8", env=env)
    elegido = r.stdout.strip()
    if not elegido:
        return actual, gr.File()          # canceló: no cambia nada
    return str(Path(elegido)), None       # limpia el archivo subido, si había


def info_video(ruta):
    """(duración, ancho, alto) con ffprobe, o None si no es un video legible."""
    if not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration", "-of", "json", ruta],
            capture_output=True, text=True, timeout=30,
        )
        d = json.loads(r.stdout)
        if not d.get("streams"):
            return None
        st = d["streams"][0]
        return float(d["format"].get("duration", 0)), st.get("width"), st.get("height")
    except Exception:
        return None


def vista_previa(archivo, ruta):
    """Miniatura del video elegido. También oculta el aviso de 'ya procesado'."""
    panel_oculto = gr.Group(visible=False)
    oculto = gr.Image(value=None, visible=False)
    entrada = resolver_entrada(archivo, ruta, estricto=False)
    if not entrada or Path(entrada).suffix.lower() not in EXT_VIDEO or not shutil.which("ffmpeg"):
        return oculto, panel_oculto

    datos = info_video(entrada)
    dur = datos[0] if datos else 0
    t = min(dur * 0.1, 60) if dur else 1   # evita la pantalla negra del arranque
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", entrada, "-frames:v", "1",
             "-vf", "scale=480:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True, timeout=60,
        )
        if r.returncode or not r.stdout:
            return oculto, panel_oculto
        img = Image.open(io.BytesIO(r.stdout))
    except Exception:
        return oculto, panel_oculto

    etiqueta = "Vista previa"
    if datos and datos[1]:
        etiqueta += f" ({hms(dur)}, {datos[1]}x{datos[2]})"
    return gr.Image(value=img, label=etiqueta, visible=True), panel_oculto


# ---------- Nombres de salida y conflictos ----------

def ruta_video_salida(stem, entrada):
    ext_out = ".mp4" if Path(entrada).suffix.lower() in EXT_MP4 else ".mkv"
    return SALIDAS / f"{stem}_subtitulado{ext_out}"


def previstos(stem, entrada, formatos, hacer_video):
    """Archivos que va a generar esta transcripción."""
    rutas = [SALIDAS / f"{stem}{SUFIJOS[f]}" for f in formatos]
    if hacer_video:
        rutas.append(ruta_video_salida(stem, entrada))
    return rutas


def nombre_libre(stem, entrada, formatos, hacer_video):
    n = 2
    while any(p.exists() for p in previstos(f"{stem}_{n}", entrada, formatos, hacer_video)):
        n += 1
    return f"{stem}_{n}"


def puede_hacer_video(entrada, subtitular):
    return bool(subtitular and Path(entrada).suffix.lower() in EXT_VIDEO and shutil.which("ffmpeg"))


def preparar(archivo, ruta, formatos, subtitular):
    """Valida todo antes de transcribir y detecta si ya existe una transcripción."""
    entrada = resolver_entrada(archivo, ruta)

    if subtitular:
        if Path(entrada).suffix.lower() not in EXT_VIDEO:
            gr.Warning("El archivo es solo audio: se transcribe igual, pero no hay video para subtitular.")
        elif not shutil.which("ffmpeg"):
            raise gr.Error(
                "No encuentro ffmpeg. Instalalo con «winget install Gyan.FFmpeg», "
                "cerrá y abrí la terminal, y volvé a ejecutar el script."
            )
    hacer_video = puede_hacer_video(entrada, subtitular)
    if not formatos and not hacer_video:
        raise gr.Error("Elegí al menos un formato de salida.")

    stem = Path(entrada).stem
    existentes = [p.name for p in previstos(stem, entrada, formatos, hacer_video) if p.exists()]
    if existentes:
        copia = nombre_libre(stem, entrada, formatos, hacer_video)
        lista = "\n".join(f"- {n}" for n in existentes)
        aviso = (f"**Este archivo ya fue procesado.** Ya existen:\n\n{lista}\n\n"
                 f"Podés sobrescribirlos o guardar los nuevos como **{copia}**.")
        return gr.Group(visible=True), aviso, None   # None = esperar decisión
    return gr.Group(visible=False), "", "nuevo"


# ---------- Transcripción ----------

def cargar_modelo(nombre):
    """Carga el modelo solo si cambió respecto al anterior."""
    if _cache["nombre"] != nombre:
        _cache["modelo"] = None  # libera la VRAM del modelo anterior
        _cache["modelo"] = WhisperModel(nombre, device="cuda", compute_type="float16")
        _cache["nombre"] = nombre
    return _cache["modelo"]


def llamar(fn, audio, **kwargs):
    """Pasa solo los parámetros que esa versión de faster-whisper acepta."""
    params = inspect.signature(fn).parameters
    return fn(audio, **{k: v for k, v in kwargs.items() if k in params and v is not None})


def palabras_de(segs):
    """Palabras con su tiempo. Si un segmento no las trae, reparte su duración
    en proporción al largo de cada palabra."""
    for s in segs:
        if s.get("palabras"):
            yield from s["palabras"]
            continue
        tokens = s["text"].split()
        if not tokens:
            continue
        total = sum(len(t) + 1 for t in tokens)
        dur = s["end"] - s["start"]
        t = s["start"]
        for tok in tokens:
            d = dur * (len(tok) + 1) / total
            yield (t, t + d, tok)
            t += d


def envolver(texto):
    """Parte el texto en dos líneas equilibradas si no entra en una."""
    if len(texto) <= SUB_CARACTERES:
        return texto
    medio = len(texto) // 2
    espacios = [i for i, c in enumerate(texto) if c == " "]
    if not espacios:
        return texto
    corte = min(espacios, key=lambda i: abs(i - medio))
    return texto[:corte] + "\n" + texto[corte + 1:]


def armar_subtitulos(segs):
    """Agrupa palabras en subtítulos cortos y legibles."""
    limite = SUB_CARACTERES * SUB_LINEAS
    subs, actual = [], []

    def cerrar():
        if actual:
            subs.append({"start": actual[0][0], "end": actual[-1][1],
                         "text": envolver(" ".join(w for _, _, w in actual))})
            actual.clear()

    for ini, fin, w in palabras_de(segs):
        w = w.strip()
        if not w:
            continue
        if actual:
            largo = len(" ".join(x[2] for x in actual)) + 1 + len(w)
            if (largo > limite
                    or fin - actual[0][0] > SUB_DURACION
                    or ini - actual[-1][1] > SUB_PAUSA):
                cerrar()
        actual.append((ini, fin, w))
        if w[-1] in ".?!…" and len(" ".join(x[2] for x in actual)) >= 15:
            cerrar()
    cerrar()

    # Que ninguno parpadee: mínimo en pantalla, sin pisar al siguiente
    for i, c in enumerate(subs):
        if c["end"] - c["start"] < SUB_MINIMO:
            tope = subs[i + 1]["start"] if i + 1 < len(subs) else c["start"] + SUB_MINIMO
            c["end"] = max(c["end"], min(c["start"] + SUB_MINIMO, tope))
    return subs


def srt_texto(segs):
    bloques = [
        f"{i}\n{tiempo_sub(s['start'], ',')} --> {tiempo_sub(s['end'], ',')}\n{s['text']}\n"
        for i, s in enumerate(segs, 1)
    ]
    return "\n".join(bloques)


def guardar(segs, stem, formatos, subs):
    contenido = {
        "TXT con tiempos": lambda: ("\n".join(f"[{hms(s['start'])}] {s['text']}" for s in segs), "utf-8-sig"),
        "TXT plano": lambda: (" ".join(s["text"] for s in segs), "utf-8-sig"),
        "SRT": lambda: (srt_texto(subs), "utf-8-sig"),
        "VTT": lambda: ("WEBVTT\n\n" + "\n".join(
            f"{tiempo_sub(s['start'], '.')} --> {tiempo_sub(s['end'], '.')}\n{s['text']}\n" for s in subs
        ), "utf-8"),
        "JSON": lambda: (json.dumps([{k: v for k, v in s.items() if k != "palabras"} for s in segs],
                                    ensure_ascii=False, indent=2), "utf-8"),
    }
    rutas = []
    for f in formatos:
        # Se concatena el sufijo (no with_suffix) para no cortar nombres con puntos
        p = SALIDAS / f"{stem}{SUFIJOS[f]}"
        texto, codificacion = contenido[f]()
        p.write_text(texto, encoding=codificacion)
        rutas.append(str(p))
    return rutas


def crear_video_subtitulado(video, subs, modo, duracion, idioma, salida, progress):
    """Crea una copia del video con subtítulos. No modifica el original."""
    with tempfile.TemporaryDirectory() as td:
        # El .srt temporal tiene nombre simple y se usa con ruta relativa (cwd=td):
        # así no hay problemas con los ":" de "C:\" ni con caracteres raros del nombre.
        (Path(td) / "subs.srt").write_text(srt_texto(subs), encoding="utf-8")

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
               "-progress", "pipe:1", "-i", str(video)]

        if modo == "soft":
            codec_sub = "mov_text" if salida.suffix == ".mp4" else "srt"
            cmd += ["-i", "subs.srt",
                    "-map", "0:v", "-map", "0:a?", "-map", "1:0",
                    "-c", "copy", "-c:s", codec_sub,
                    "-metadata:s:s:0", f"language={LANG3.get(idioma, 'und')}",
                    "-disposition:s:0", "default"]
        else:
            estilo = "FontSize=16,Outline=1,Shadow=0,MarginV=25"
            cmd += ["-vf", f"subtitles=subs.srt:force_style='{estilo}'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "copy"]

        cmd.append(str(salida))

        proc = subprocess.Popen(
            cmd, cwd=td, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        for linea in proc.stdout:
            clave, _, valor = linea.strip().partition("=")
            if clave in ("out_time_us", "out_time_ms"):  # ambos vienen en microsegundos
                try:
                    seg = int(valor) / 1_000_000
                except ValueError:
                    continue
                seg = max(seg, 0)  # ffmpeg reporta tiempos negativos al arrancar
                progress(min(seg / duracion, 1), desc=f"Creando video: {hms(seg)} de {hms(duracion)}")
        error = proc.stderr.read()
        proc.wait()

    # Si había subtítulos extraídos para la vista previa, quedaron viejos
    (VISTA / f"{salida.stem}.vtt").unlink(missing_ok=True)

    if proc.returncode != 0:
        salida.unlink(missing_ok=True)
        raise gr.Error(f"ffmpeg falló al crear el video: {error.strip()[-400:]}")

    return salida


def transcribir(archivo, ruta, modelo, idioma, vocab, rapido, lote, formatos,
                subtitular, modo_sub, decision, progress=gr.Progress()):
    if decision is None:
        # Hay un conflicto de nombres esperando respuesta: no hacer nada
        return gr.File(), gr.Textbox()

    entrada = resolver_entrada(archivo, ruta)
    hacer_video = puede_hacer_video(entrada, subtitular)

    stem = Path(entrada).stem
    if decision == "renombrar":
        stem = nombre_libre(stem, entrada, formatos, hacer_video)

    progress(0, desc=f"Cargando {modelo} (la primera vez se descarga)")
    m = cargar_modelo(modelo)

    vocab = (vocab or "").strip()
    necesita_subs = hacer_video or "SRT" in formatos or "VTT" in formatos
    opciones = dict(
        language=IDIOMAS[idioma],
        vad_filter=True,
        beam_size=5,
        initial_prompt=f"Vocabulario: {vocab}." if vocab else None,
        hotwords=vocab or None,
        without_timestamps=False,          # en modo por lotes el default es True: segmentos de 30 s
        word_timestamps=necesita_subs,     # tiempo de cada palabra, para cortar bien los subtítulos
    )
    if rapido:
        motor = BatchedInferencePipeline(model=m)
        opciones["batch_size"] = int(lote)
    else:
        motor = m

    progress(0, desc="Analizando el audio")
    segs = []
    try:
        segments, info = llamar(motor.transcribe, entrada, **opciones)
        total = max(info.duration, 1)
        for s in segments:
            seg = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            if getattr(s, "words", None):
                seg["palabras"] = [(w.start, w.end, w.word) for w in s.words]
            segs.append(seg)
            progress(min(s.end / total, 1), desc=f"Transcribiendo {hms(s.end)} de {hms(total)}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            consejo = ("bajá el tamaño de lote" if rapido
                       else "probá con large-v3-turbo o un modelo más chico")
            raise gr.Error(f"La GPU se quedó sin memoria: {consejo}.")
        raise

    subs = armar_subtitulos(segs) if necesita_subs else []
    rutas = guardar(segs, stem, formatos, subs)

    modo_txt = f"rápido, lote {int(lote)}" if rapido else "normal"
    resumen = (f"Idioma: {info.language}   Duración: {hms(info.duration)}   "
               f"Segmentos: {len(segs)}   Modo: {modo_txt}\n")
    resumen += f"Guardado en: {SALIDAS}"
    resumen += f" como «{stem}»\n" if decision == "renombrar" else "\n"

    if hacer_video:
        video = crear_video_subtitulado(entrada, subs, MODOS_SUB[modo_sub], total, info.language,
                                        ruta_video_salida(stem, entrada), progress)
        resumen += f"Video con subtítulos: {video.name}  (está en la pestaña «Archivos guardados»)\n"

    vista = "\n".join(f"[{hms(s['start'])}] {s['text']}" for s in segs)
    return rutas, resumen + "\n" + vista


# ---------- Archivos guardados ----------

def opciones_archivos(filtro="Todos"):
    """Lista de (etiqueta visible, nombre de archivo), más nuevos primero."""
    exts = FILTROS.get(filtro)
    archivos = sorted(
        (p for p in SALIDAS.iterdir()
         if p.is_file() and (exts is None or p.suffix.lower() in exts)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [(f"{p.name}   ({tamano(p.stat().st_size)}, {fecha(p)})", p.name) for p in archivos]


def ruta_segura(nombre):
    """Evita que se toque cualquier cosa fuera de la carpeta salidas."""
    if not nombre:
        return None
    p = (SALIDAS / Path(nombre).name).resolve()
    return p if p.parent == SALIDAS.resolve() and p.is_file() else None


def refrescar(filtro="Todos"):
    return gr.CheckboxGroup(choices=opciones_archivos(filtro), value=[])


def subtitulos_incrustados(video):
    """Extrae la pista de subtítulos (si la hay) a .vtt para mostrarla en el navegador."""
    if not shutil.which("ffmpeg"):
        return None
    VISTA.mkdir(exist_ok=True)
    destino = VISTA / f"{video.stem}.vtt"
    if destino.exists() and destino.stat().st_mtime >= video.stat().st_mtime:
        return str(destino)   # ya extraído y al día
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:s:0", "-f", "webvtt", str(destino)],
        capture_output=True, timeout=300,
    )
    if r.returncode == 0 and destino.exists() and destino.stat().st_size > 10:
        return str(destino)
    destino.unlink(missing_ok=True)
    return None   # sin pista de subtítulos (por ejemplo, grabados en la imagen)


def mostrar(nombre):
    """Devuelve (título, vista de texto, vista de video, archivo actual)."""
    sin_texto = gr.Textbox(value="", visible=False)
    sin_video = gr.Video(value=None, visible=False)
    p = ruta_segura(nombre)
    if not p:
        return "*Marcá un archivo de la lista para verlo acá.*", sin_texto, sin_video, None

    titulo = f"**{p.name}**  \n{tamano(p.stat().st_size)}, modificado el {fecha(p)}"
    ext = p.suffix.lower()

    if ext in EXT_TEXTO:
        texto = p.read_text(encoding="utf-8-sig", errors="replace")
        if len(texto) > LIMITE_TEXTO:
            texto = texto[:LIMITE_TEXTO]
            titulo += f"  \nSe muestran los primeros {LIMITE_TEXTO:,} caracteres."
        return titulo, gr.Textbox(value=texto, visible=True), sin_video, p.name

    if ext in EXT_NAVEGADOR:
        subs = subtitulos_incrustados(p)
        valor = (str(p), subs) if subs else str(p)
        return titulo, sin_texto, gr.Video(value=valor, visible=True), p.name

    if ext in EXT_VIDEO:
        titulo += "\n\nEl navegador no reproduce este formato. Usá «Abrir con el programa predeterminado»."
    else:
        titulo += "\n\nNo hay vista previa para este tipo de archivo."
    return titulo, sin_texto, sin_video, p.name


def al_marcar(seleccion, evt: gr.SelectData):
    """Muestra el archivo recién marcado; si se desmarcó, el último que quede marcado."""
    nombre = evt.value if evt.selected else (seleccion[-1] if seleccion else None)
    if nombre and not ruta_segura(nombre):
        # Por si la versión de Gradio devuelve la etiqueta en vez del valor
        nombre = next((v for etiqueta, v in opciones_archivos() if etiqueta == nombre), None)
    return mostrar(nombre)


def abrir(nombre):
    p = ruta_segura(nombre)
    if not p:
        raise gr.Error("Marcá un archivo primero.")
    os.startfile(p)


def mostrar_en_carpeta(nombre):
    p = ruta_segura(nombre)
    if p:
        subprocess.Popen(["explorer", f"/select,{p}"])
    else:
        os.startfile(SALIDAS)


def descargar(seleccion):
    if not seleccion:
        raise gr.Error("Seleccioná al menos un archivo.")
    return [str(p) for n in seleccion if (p := ruta_segura(n))]


def eliminar(seleccion, actual, filtro):
    sin_cambios = (gr.Markdown(), gr.Textbox(), gr.Video(), actual)
    if not seleccion:
        gr.Info("No se eliminó nada.")
        return (refrescar(filtro), None) + sin_cambios
    borrados = 0
    for n in seleccion:
        p = ruta_segura(n)
        if p:
            p.unlink()
            (VISTA / f"{p.stem}.vtt").unlink(missing_ok=True)
            borrados += 1
    gr.Info(f"Se eliminaron {borrados} archivo(s).")
    vista = mostrar(None) if actual in seleccion else sin_cambios
    return (refrescar(filtro), None) + vista


# ---------- Interfaz ----------

with gr.Blocks() as demo:
    gr.Markdown(f"## Transcriptor Whisper v{VERSION}")
    decision = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Transcribir"):
            with gr.Row():
                with gr.Column(scale=1):
                    archivo = gr.File(label="Subir video o audio", file_types=["audio", "video"], type="filepath")
                    ruta = gr.Textbox(
                        label="…o ruta en esta PC (no copia el archivo; tiene prioridad)",
                        placeholder=r"D:\Descargas\clase_git.mp4",
                    )
                    b_examinar = gr.Button("Examinar…", size="sm")
                    preview = gr.Image(label="Vista previa", type="pil", interactive=False,
                                       visible=False, height=220)

                    modelo = gr.Dropdown(list(MODELOS), value="large-v3", label="Modelo")
                    info_modelo = gr.Markdown(MODELOS["large-v3"])
                    idioma = gr.Dropdown(list(IDIOMAS), value="Español", label="Idioma")
                    vocab = gr.Textbox(
                        label="Vocabulario (opcional)",
                        placeholder="git, commit, rebase, merge, GitHub, pull request",
                        info="Nombres propios y términos técnicos que aparecen en el audio, "
                             "separados por comas. Ayuda a que Whisper los escriba bien.",
                    )
                    rapido = gr.Checkbox(
                        value=True, label="Modo rápido (por lotes)",
                        info="Procesa varios fragmentos a la vez en la GPU. Suele ser 3 a 4 veces más rápido.",
                    )
                    lote = gr.Slider(
                        2, 32, value=8, step=2, label="Tamaño de lote",
                        info="Más alto: más rápido, pero usa más VRAM. Si da error de memoria, bajalo.",
                    )

                    formatos = gr.CheckboxGroup(
                        FORMATOS, value=["TXT con tiempos", "TXT plano"], label="Formatos de salida"
                    )
                    subtitular = gr.Checkbox(label="Crear copia del video con subtítulos")
                    modo_sub = gr.Radio(
                        list(MODOS_SUB), value=list(MODOS_SUB)[0],
                        label="Tipo de subtítulos", visible=False,
                    )
                    boton = gr.Button("Transcribir", variant="primary")

                    with gr.Group(visible=False) as panel:
                        aviso = gr.Markdown()
                        with gr.Row():
                            b_sobrescribir = gr.Button("Sobrescribir", variant="stop")
                            b_renombrar = gr.Button("Guardar como copia", variant="primary")
                            b_cancelar = gr.Button("Cancelar")

                with gr.Column(scale=2):
                    descargas = gr.File(label="Archivos generados", file_count="multiple")
                    vista = gr.Textbox(label="Transcripción", lines=25)

        with gr.Tab("Archivos guardados") as tab_archivos:
            actual = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    filtro = gr.Radio(list(FILTROS), value="Todos", label="Mostrar")
                    lista = gr.CheckboxGroup(
                        choices=opciones_archivos(),
                        label="Marcá un archivo para verlo; marcá varios para descargar o eliminar",
                    )
                    with gr.Row():
                        b_refrescar = gr.Button("Actualizar lista", size="sm")
                        b_descargar = gr.Button("Descargar seleccionados", size="sm", variant="primary")
                        b_eliminar = gr.Button("Eliminar seleccionados", size="sm", variant="stop")
                    archivos_descarga = gr.File(label="Listos para descargar", file_count="multiple")

                with gr.Column(scale=2):
                    titulo_prev = gr.Markdown("*Marcá un archivo de la lista para verlo acá.*")
                    with gr.Row():
                        b_abrir = gr.Button("Abrir con el programa predeterminado", size="sm")
                        b_carpeta = gr.Button("Mostrar en la carpeta", size="sm")
                    prev_texto = gr.Textbox(label="Contenido", lines=25, visible=False)
                    prev_video = gr.Video(label="Video", visible=False, interactive=False)

    # --- Eventos: entrada y vista previa del video
    archivo.upload(lambda: "", None, ruta).then(vista_previa, [archivo, ruta], [preview, panel])
    archivo.clear(vista_previa, [archivo, ruta], [preview, panel])
    ruta.submit(vista_previa, [archivo, ruta], [preview, panel])
    ruta.blur(vista_previa, [archivo, ruta], [preview, panel])
    b_examinar.click(examinar, ruta, [ruta, archivo]).then(vista_previa, [archivo, ruta], [preview, panel])

    # --- Eventos: opciones
    modelo.change(lambda m: MODELOS[m], modelo, info_modelo)
    rapido.change(lambda v: gr.Slider(visible=v), rapido, lote)
    subtitular.change(lambda v: gr.Radio(visible=v), subtitular, modo_sub)

    # --- Eventos: transcribir (con chequeo de archivos existentes)
    entradas = [archivo, ruta, modelo, idioma, vocab, rapido, lote, formatos, subtitular, modo_sub, decision]
    salidas_tr = [descargas, vista]

    boton.click(
        preparar, [archivo, ruta, formatos, subtitular], [panel, aviso, decision]
    ).success(transcribir, entradas, salidas_tr).then(refrescar, filtro, lista)

    b_sobrescribir.click(
        lambda: (gr.Group(visible=False), "sobrescribir"), None, [panel, decision]
    ).then(transcribir, entradas, salidas_tr).then(refrescar, filtro, lista)

    b_renombrar.click(
        lambda: (gr.Group(visible=False), "renombrar"), None, [panel, decision]
    ).then(transcribir, entradas, salidas_tr).then(refrescar, filtro, lista)

    b_cancelar.click(lambda: gr.Group(visible=False), None, panel)

    # --- Eventos: archivos guardados
    vista_arch = [titulo_prev, prev_texto, prev_video, actual]
    tab_archivos.select(refrescar, filtro, lista)
    filtro.change(refrescar, filtro, lista)
    b_refrescar.click(refrescar, filtro, lista)
    lista.select(al_marcar, lista, vista_arch)
    b_abrir.click(abrir, actual, None)
    b_carpeta.click(mostrar_en_carpeta, actual, None)
    b_descargar.click(descargar, lista, archivos_descarga)
    b_eliminar.click(
        eliminar,
        [lista, actual, filtro],
        [lista, archivos_descarga] + vista_arch,
        # Pide confirmación en el navegador; si cancelás, no se borra nada
        js="(sel, act, f) => [(sel.length && confirm(`¿Eliminar ${sel.length} archivo(s)? No se puede deshacer.`)) ? sel : [], act, f]",
    )
    demo.load(refrescar, filtro, lista)

if __name__ == "__main__":
    demo.launch(inbrowser=True, allowed_paths=[str(SALIDAS)])