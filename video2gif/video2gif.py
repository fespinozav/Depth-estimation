#!/usr/bin/env python3
import argparse, shutil, subprocess, sys, os

def have_ffmpeg():
    return shutil.which("ffmpeg") is not None

def build_cmd(args):
    fps = args.fps
    scale = f"scale={args.width}:-1:flags=lanczos" if args.width else "scale=iw:ih"
    loop = "loop=0"  # loops forever (GIF)

    if args.transparent:
        if args.has_alpha:
            # Preserva alpha existente
            filter_complex = (
                f"[0:v]fps={fps},{scale},format=rgba,split[v1][v2];"
                f"[v1]palettegen=max_colors=255:reserve_transparent=1[p];"
                f"[v2][p]paletteuse=dither=bayer:bayer_scale=5:alpha_threshold=128"
            )
        else:
            # Chroma key: vuelve transparente el color indicado
            key = args.keycolor.lower()
            # ffmpeg acepta 0xRRGGBB
            if key.startswith("#"): key = "0x" + key[1:]
            filter_complex = (
                f"[0:v]fps={fps},{scale},"
                f"chromakey={key}:{args.similarity}:{args.blend},"
                f"format=rgba,split[v1][v2];"
                f"[v1]palettegen=max_colors=255:reserve_transparent=1[p];"
                f"[v2][p]paletteuse=dither=bayer:bayer_scale=5:alpha_threshold=128"
            )
    else:
        # GIF normal (sin transparencia)
        filter_complex = (
            f"[0:v]fps={fps},{scale},split[v1][v2];"
            f"[v1]palettegen=stats_mode=diff[p];"
            f"[v2][p]paletteuse=dither=floyd_steinberg"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", args.input,
        "-filter_complex", filter_complex,
        "-gifflags", loop,
    ]

    # Control de calidad/opciones extra
    if args.lossy:
        # Reduce tamaño con dithering Bayer + menos colores
        cmd[-2:-2] = ["-final_delay", "0"]  # evita pausas al final (algunos viewers)
    if args.output.lower().endswith(".gif") is False:
        print("Advertencia: la salida no termina en .gif; la ajustaré automáticamente.")
        args.output += ".gif"

    cmd += [args.output]
    return cmd

def main():
    ap = argparse.ArgumentParser(description="Video a GIF (con opción de transparencia). Requiere ffmpeg.")
    ap.add_argument("input", help="Ruta del video de entrada (mp4/mov/etc.)")
    ap.add_argument("output", help="Ruta del GIF de salida (ej: salida.gif)")
    ap.add_argument("--fps", type=int, default=12, help="FPS del GIF")
    ap.add_argument("--width", type=int, default=512, help="Ancho del GIF (mantiene aspecto). 0 = sin cambio")
    ap.add_argument("--lossy", action="store_true", help="GIF más liviano (puede perder calidad)")
    # Transparencia
    ap.add_argument("--transparent", action="store_true", help="Habilitar fondo transparente")
    ap.add_argument("--has-alpha", action="store_true", help="El video YA tiene canal alpha (preservarlo)")
    ap.add_argument("--keycolor", default="#00FF00", help="Color a volver transparente (hex) si no hay alpha")
    ap.add_argument("--similarity", type=float, default=0.12, help="Umbral de similitud (0..1) para chroma key")
    ap.add_argument("--blend", type=float, default=0.0, help="Mezcla de bordes para chroma key (0..1)")
    args = ap.parse_args()

    if not have_ffmpeg():
        print("Error: ffmpeg no está instalado o no está en PATH.")
        print("macOS: brew install ffmpeg | Ubuntu: sudo apt-get install ffmpeg | Windows: choco install ffmpeg")
        sys.exit(1)

    if args.width == 0:
        args.width = None

    cmd = build_cmd(args)
    try:
        subprocess.run(cmd, check=True)
        print(f"OK: GIF generado en {args.output}")
    except subprocess.CalledProcessError as e:
        print("ffmpeg falló.\nComando:", " ".join(cmd))
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()