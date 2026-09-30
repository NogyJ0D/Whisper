"""
Transcriptor Whisper — v1.3

v1.3
  - Opción "Crear copia del video con subtítulos" (usa la transcripción con tiempos)
      * Incrustados: rápido, sin re-codificar; se activan y desactivan en el reproductor
      * Grabados en la imagen: lento, re-codifica; se ven en cualquier reproductor
  - El video original no se toca: la copia queda en "salidas" como <nombre>_subtitulado
  - Requiere ffmpeg instalado; se verifica ANTES de transcribir para no perder tiempo

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

import os
import json
import shutil
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
from faster_whisper import WhisperModel

VERSION = "1.3"

SALIDAS = Path(__file__).parent / "salidas"
SALIDAS.mkdir(exist_ok=True)

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
FORMATOS = ["TXT con tiempos", "TXT plano", "SRT", "VTT", "JSON"]

EXT_VIDEO = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".ts"}
EXT_MP4 = {".mp4", ".m4v", ".mov"}
MODOS_SUB = {
    "Incrustados (rápido, se activan y desactivan en el reproductor)": "soft",
    "Grabados en la imagen (lento, se ven en cualquier reproductor)": "hard",
}
LANG3 = {"es": "spa", "en": "eng"}

_cache = {"nombre": None, "modelo": None}


# ---------- Transcripción ----------

def cargar_modelo(nombre):
    """Carga el modelo solo si cambió respecto al anterior."""
    if _cache["nombre"] != nombre:
        _cache["modelo"] = None  # libera la VRAM del modelo anterior
        _cache["modelo"] = WhisperModel(nombre, device="cuda", compute_type="float16")
        _cache["nombre"] = nombre
    return _cache["modelo"]


def hms(t):
    """00:05:01"""
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"


def tiempo_sub(t, sep):
    """00:05:01,250 (SRT) o 00:05:01.250 (VTT)"""
    return f"{hms(t)}{sep}{int((t % 1) * 1000):03}"


def srt_texto(segs):
    bloques = [
        f"{i}\n{tiempo_sub(s['start'], ',')} --> {tiempo_sub(s['end'], ',')}\n{s['text']}\n"
        for i, s in enumerate(segs, 1)
    ]
    return "\n".join(bloques)


def guardar(segs, base, formatos):
    rutas = []

    if "TXT con tiempos" in formatos:
        p = base.with_name(base.name + "_tiempos.txt")
        p.write_text(
            "\n".join(f"[{hms(s['start'])}] {s['text']}" for s in segs),
            encoding="utf-8-sig",
        )
        rutas.append(str(p))

    if "TXT plano" in formatos:
        p = base.with_suffix(".txt")
        p.write_text(" ".join(s["text"] for s in segs), encoding="utf-8-sig")
        rutas.append(str(p))

    if "SRT" in formatos:
        p = base.with_suffix(".srt")
        p.write_text(srt_texto(segs), encoding="utf-8-sig")
        rutas.append(str(p))

    if "VTT" in formatos:
        p = base.with_suffix(".vtt")
        bloques = [
            f"{tiempo_sub(s['start'], '.')} --> {tiempo_sub(s['end'], '.')}\n{s['text']}\n"
            for s in segs
        ]
        p.write_text("WEBVTT\n\n" + "\n".join(bloques), encoding="utf-8")
        rutas.append(str(p))

    if "JSON" in formatos:
        p = base.with_suffix(".json")
        p.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        rutas.append(str(p))

    return rutas


def crear_video_subtitulado(video, segs, modo, duracion, idioma, progress):
    """Crea una copia del video con subtítulos. No modifica el original."""
    ext = Path(video).suffix.lower()
    ext_out = ".mp4" if ext in EXT_MP4 else ".mkv"
    salida = SALIDAS / f"{Path(video).stem}_subtitulado{ext_out}"

    with tempfile.TemporaryDirectory() as td:
        # El .srt temporal tiene nombre simple y se usa con ruta relativa (cwd=td):
        # así no hay problemas con los ":" de "C:\" ni con caracteres raros del nombre.
        (Path(td) / "subs.srt").write_text(srt_texto(segs), encoding="utf-8")

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
               "-progress", "pipe:1", "-i", str(video)]

        if modo == "soft":
            codec_sub = "mov_text" if ext_out == ".mp4" else "srt"
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

    if proc.returncode != 0:
        salida.unlink(missing_ok=True)
        raise gr.Error(f"ffmpeg falló al crear el video: {error.strip()[-400:]}")

    return salida


def transcribir(archivo, modelo, idioma, formatos, subtitular, modo_sub, progress=gr.Progress()):
    if not archivo:
        raise gr.Error("Elegí un archivo de audio o video.")

    # Validaciones ANTES de transcribir, para no descubrir el problema después de una hora
    hacer_video = False
    if subtitular:
        if Path(archivo).suffix.lower() not in EXT_VIDEO:
            gr.Warning("El archivo es solo audio: se transcribe igual, pero no hay video para subtitular.")
        elif not shutil.which("ffmpeg"):
            raise gr.Error(
                "No encuentro ffmpeg. Instalalo con «winget install Gyan.FFmpeg», "
                "cerrá y abrí la terminal, y volvé a ejecutar el script."
            )
        else:
            hacer_video = True
    if not formatos and not hacer_video:
        raise gr.Error("Elegí al menos un formato de salida.")

    progress(0, desc=f"Cargando {modelo} (la primera vez se descarga)")
    m = cargar_modelo(modelo)

    progress(0, desc="Analizando el audio")
    segments, info = m.transcribe(
        archivo,
        language=IDIOMAS[idioma],
        vad_filter=True,
        beam_size=5,
    )

    total = max(info.duration, 1)
    segs = []
    for s in segments:
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        progress(min(s.end / total, 1), desc=f"Transcribiendo {hms(s.end)} de {hms(total)}")

    base = SALIDAS / Path(archivo).stem
    rutas = guardar(segs, base, formatos)

    resumen = f"Idioma: {info.language}   Duración: {hms(info.duration)}   Segmentos: {len(segs)}\n"
    resumen += f"Guardado en: {SALIDAS}\n"

    if hacer_video:
        video = crear_video_subtitulado(archivo, segs, MODOS_SUB[modo_sub], total, info.language, progress)
        resumen += f"Video con subtítulos: {video.name}  (está en la pestaña «Archivos guardados»)\n"

    vista = "\n".join(f"[{hms(s['start'])}] {s['text']}" for s in segs)
    return rutas, resumen + "\n" + vista


# ---------- Archivos guardados ----------

def tamano(bytes_):
    for unidad in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.0f} {unidad}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def opciones_archivos():
    """Lista de (etiqueta visible, nombre de archivo), más nuevos primero."""
    archivos = sorted(
        (p for p in SALIDAS.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    opciones = []
    for p in archivos:
        st = p.stat()
        fecha = datetime.fromtimestamp(st.st_mtime).strftime("%d/%m/%Y %H:%M")
        opciones.append((f"{p.name}   ({tamano(st.st_size)}, {fecha})", p.name))
    return opciones


def ruta_segura(nombre):
    """Evita que se toque cualquier cosa fuera de la carpeta salidas."""
    p = (SALIDAS / Path(nombre).name).resolve()
    return p if p.parent == SALIDAS.resolve() and p.is_file() else None


def refrescar():
    return gr.CheckboxGroup(choices=opciones_archivos(), value=[])


def descargar(seleccion):
    if not seleccion:
        raise gr.Error("Seleccioná al menos un archivo.")
    rutas = [str(p) for n in seleccion if (p := ruta_segura(n))]
    return rutas


def eliminar(seleccion):
    if not seleccion:
        gr.Info("No se eliminó nada.")
        return refrescar(), None
    borrados = 0
    for n in seleccion:
        p = ruta_segura(n)
        if p:
            p.unlink()
            borrados += 1
    gr.Info(f"Se eliminaron {borrados} archivo(s).")
    return refrescar(), None


# ---------- Interfaz ----------

with gr.Blocks() as demo:
    gr.Markdown(f"## Transcriptor Whisper v{VERSION}")

    with gr.Tabs():
        with gr.Tab("Transcribir"):
            with gr.Row():
                with gr.Column(scale=1):
                    archivo = gr.File(label="Video o audio", file_types=["audio", "video"], type="filepath")
                    modelo = gr.Dropdown(list(MODELOS), value="large-v3", label="Modelo")
                    info_modelo = gr.Markdown(MODELOS["large-v3"])
                    idioma = gr.Dropdown(list(IDIOMAS), value="Español", label="Idioma")
                    formatos = gr.CheckboxGroup(
                        FORMATOS, value=["TXT con tiempos", "TXT plano"], label="Formatos de salida"
                    )
                    subtitular = gr.Checkbox(label="Crear copia del video con subtítulos")
                    modo_sub = gr.Radio(
                        list(MODOS_SUB), value=list(MODOS_SUB)[0],
                        label="Tipo de subtítulos", visible=False,
                    )
                    boton = gr.Button("Transcribir", variant="primary")

                with gr.Column(scale=2):
                    descargas = gr.File(label="Archivos generados", file_count="multiple")
                    vista = gr.Textbox(label="Transcripción", lines=25)

        with gr.Tab("Archivos guardados"):
            lista = gr.CheckboxGroup(choices=opciones_archivos(), label=f"Archivos en {SALIDAS}")
            with gr.Row():
                b_refrescar = gr.Button("Actualizar lista")
                b_descargar = gr.Button("Descargar seleccionados", variant="primary")
                b_eliminar = gr.Button("Eliminar seleccionados", variant="stop")
            archivos_descarga = gr.File(label="Listos para descargar", file_count="multiple")

    # Eventos
    modelo.change(lambda m: MODELOS[m], modelo, info_modelo)
    subtitular.change(lambda v: gr.Radio(visible=v), subtitular, modo_sub)

    boton.click(
        transcribir, [archivo, modelo, idioma, formatos, subtitular, modo_sub], [descargas, vista]
    ).then(refrescar, None, lista)

    b_refrescar.click(refrescar, None, lista)
    b_descargar.click(descargar, lista, archivos_descarga)
    b_eliminar.click(
        eliminar,
        lista,
        [lista, archivos_descarga],
        # Pide confirmación en el navegador; si cancelás, no se borra nada
        js="(sel) => [(sel.length && confirm(`¿Eliminar ${sel.length} archivo(s)? No se puede deshacer.`)) ? sel : []]",
    )
    demo.load(refrescar, None, lista)

if __name__ == "__main__":
    demo.launch(inbrowser=True, allowed_paths=[str(SALIDAS)])