#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch


# -----------------------------
# 0. Funciones auxiliares
# -----------------------------
def pick_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_midas(model_type: str, device: torch.device, use_fp16: bool):
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(device).eval()
    # FP16 solo en CUDA; MPS no soporta half estable
    if use_fp16 and device.type == "cuda":
        midas.half()

    tfs = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if "DPT" in model_type:
        transform = tfs.dpt_transform
    else:
        transform = tfs.small_transform
    return midas, transform


def percentile_normalize(depth: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """Escala a [0,1] usando recorte por percentiles para reducir parpadeo."""
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, [p_low, p_high])
    # Evita división por cero
    denom = max(hi - lo, 1e-6)
    d = np.clip((d - lo) / denom, 0.0, 1.0)
    return d


def apply_postprocess(depth_u8: np.ndarray, bilateral: bool) -> np.ndarray:
    if bilateral:
        # Suavizado preservando bordes
        return cv.bilateralFilter(depth_u8, d=5, sigmaColor=50, sigmaSpace=50)
    return depth_u8


def colorize(depth_u8: np.ndarray) -> np.ndarray:
    return cv.applyColorMap(depth_u8, cv.COLORMAP_MAGMA)


# -----------------------------
# 1. Lectura y encuadre al tamaño objetivo
# -----------------------------
def window_should_close(win_name: str) -> bool:
    """
    Devuelve True si la ventana fue cerrada (botón 'X') o dejó de estar visible.
    """
    try:
        return cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1
    except cv.error:
        # Si la ventana ya no existe, consideramos que debe cerrarse el bucle
        return True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# Helper para determinar el directorio final de salida según flags
def get_output_dir(args) -> Path:
    """
    Devuelve el directorio final de salida considerando:
    - args.output_dir (base)
    - args.use_model_subdir  -> añade el nombre del modelo como subcarpeta
    - args.output_subdir     -> añade una subcarpeta personalizada
    """
    base = Path(args.output_dir)
    if getattr(args, "use_model_subdir", False):
        base = base / args.model
    # output_subdir puede ser cadena vacía -> no añadir
    if getattr(args, "output_subdir", ""):
        base = base / args.output_subdir
    return base


def write_csv_header(csv_path: Path):
    new_file = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "timestamp", "source", "index", "filename",
            "model", "device", "input_size", "fp16", "bilateral",
            "width", "height", "fps"
        ])
    return f, writer


def backend_flag(backend: str):
    backend = (backend or "auto").lower()
    # Devuelve la flag de backend de OpenCV o None para auto
    if backend == "avfoundation":
        return cv.CAP_AVFOUNDATION
    if backend == "dshow":
        return cv.CAP_DSHOW
    if backend == "v4l2":
        return cv.CAP_V4L2
    return None


# -----------------------------
# 2. Inferencia con MiDaS
# -----------------------------
def run_inference(
    image_bgr: np.ndarray,
    midas,
    transform,
    device: torch.device,
    target_size: int,
    use_fp16: bool,
):
    # Convertir a RGB
    img_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    # Redimensionado “letterbox” sencillo para mantener aspecto hacia el tamaño objetivo
    h, w = img_rgb.shape[:2]
    scale = float(target_size) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv.resize(img_rgb, (nw, nh), interpolation=cv.INTER_AREA)

    # El transform de MiDaS ya normaliza; lo aplicamos sobre img_resized
    input_batch = transform(img_resized).to(device, non_blocking=(device.type == "cuda"))
    if use_fp16 and device.type == "cuda":
        input_batch = input_batch.half()

    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                pred = midas(input_batch)
        else:
            pred = midas(input_batch)

        # A la resolución original (h, w) para facilitar overlays
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze(1).squeeze(0)

    depth = pred.detach().float().cpu().numpy()
    return depth  # float32 arbitraria (relativa/inversa)


# -----------------------------
# 3.b Fuentes de entrada y procesamiento de imágenes
# -----------------------------
def process_webcam(args, midas, transform, device):
    # Backend opcional (macOS: avfoundation suele ser más estable)
    backend = backend_flag(args.backend)
    cap = cv.VideoCapture(args.cam_index if args.cam_index is not None else 0,
                          backend if backend is not None else 0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    out_dir = get_output_dir(args)
    ensure_dir(out_dir)
    csv_file, writer = write_csv_header(out_dir / "log.csv")

    ema = None
    writer_video = None
    out_path_webcam = str(out_dir / "webcam_depth_colored.mp4")
    try:
        t_prev = time.perf_counter()
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame perdido; intentando continuar…")
                continue

            t0 = time.perf_counter()
            depth = run_inference(frame, midas, transform, device, args.size, args.fp16)
            # -----------------------------
            # 4.  Normalización por percentiles
            # -----------------------------
            depth_norm = percentile_normalize(depth, args.p_low, args.p_high)
            depth_u8 = (depth_norm * 255).astype(np.uint8)
            depth_u8 = apply_postprocess(depth_u8, args.bilateral)

            # -----------------------------
            # 5.  Suavizado temporal (EMA)
            # -----------------------------
            if args.ema_alpha is not None:
                if ema is None:
                    ema = depth_u8.copy()
                else:
                    a = float(args.ema_alpha)
                    ema = (a * ema + (1.0 - a) * depth_u8).astype(np.uint8)
                vis_u8 = ema
            else:
                vis_u8 = depth_u8

            depth_color = colorize(vis_u8)

            # Mostrar lado a lado
            viz = cv.hconcat([frame, depth_color])

            # Inicializa grabación MP4 en el primer frame si se solicitó
            if args.record_webcam and writer_video is None:
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                writer_video = cv.VideoWriter(out_path_webcam, fourcc, args.webcam_fps, (viz.shape[1], viz.shape[0]))
                if not writer_video.isOpened():
                    print(f"[WARN] No se pudo abrir el VideoWriter para {out_path_webcam}")
                else:
                    print(f"[INFO] Grabando webcam a {out_path_webcam} @ {args.webcam_fps} FPS")

            # Si está activo, escribir frame al MP4
            if writer_video is not None:
                writer_video.write(viz)

            # FPS
            t1 = time.perf_counter()
            fps = 1.0 / max(t1 - t_prev, 1e-6)
            t_prev = t1

            cv.putText(
                viz,
                f"{args.model} | {device.type.upper()} | {fps:5.1f} FPS",
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if not args.no_display:
                cv.imshow("MiDaS Depth (webcam)", viz)
                # Cerrar si el usuario presiona el botón 'X' de la ventana
                if window_should_close("MiDaS Depth (webcam)"):
                    break

            # Guardado por frame (opcional)
            if args.save_every_n > 0 and (idx % args.save_every_n == 0):
                base = f"webcam_{idx:06d}"
                save_artifacts(depth, depth_u8, depth_color, out_dir, base, args)

            # Log CSV
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "webcam",
                idx,
                "",
                args.model,
                device.type,
                args.size,
                int(args.fp16 and device.type == "cuda"),
                int(args.bilateral),
                frame.shape[1],
                frame.shape[0],
                f"{fps:.3f}",
            ])
            csv_file.flush()

            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break

            key = cv.waitKey(1) & 0xFF
            # Permitir salir con 'q', 'x' o ESC (27)
            if key in (ord('q'), ord('x'), 27):
                break
            if key == ord('s'):
                base = f"snap_{int(time.time())}"
                save_artifacts(depth, depth_u8, depth_color, out_dir, base, args)

    finally:
        cap.release()
        if writer_video is not None:
            writer_video.release()
            print(f"[INFO] Video guardado en {out_path_webcam}")
        cv.destroyAllWindows()
        csv_file.close()

# -----------------------------
# 3.b Fuentes de entrada y procesamiento de imágenes (video)
# -----------------------------
def process_video(args, midas, transform, device):
    backend = backend_flag(args.backend)
    cap = cv.VideoCapture(args.path, backend if backend is not None else 0)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.path}")

    out_dir = get_output_dir(args)
    ensure_dir(out_dir)
    csv_file, writer = write_csv_header(out_dir / "log.csv")

    # Preparar escritor de video si se solicita
    writer_video = None
    if args.save_video:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out_path = str(out_dir / "depth_colored.mp4")
        fps_in = cap.get(cv.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer_video = cv.VideoWriter(out_path, fourcc, fps_in, (w * 2, h))  # original|depth
        if not writer_video.isOpened():
            print(f"[WARN] No se pudo abrir el VideoWriter para {out_path}")
        else:
            print(f"[INFO] Grabando video procesado a {out_path} @ {fps_in} FPS")

    ema = None
    try:
        idx = 0
        t_prev = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            depth = run_inference(frame, midas, transform, device, args.size, args.fp16)

            depth_norm = percentile_normalize(depth, args.p_low, args.p_high)
            depth_u8 = (depth_norm * 255).astype(np.uint8)
            depth_u8 = apply_postprocess(depth_u8, args.bilateral)

            if args.ema_alpha is not None:
                if ema is None:
                    ema = depth_u8.copy()
                else:
                    a = float(args.ema_alpha)
                    # Filtro EMA por píxel entre cuadros consecutivos
                    ema = (a * ema + (1.0 - a) * depth_u8).astype(np.uint8)
                vis_u8 = ema
            else:
                vis_u8 = depth_u8

            depth_color = colorize(vis_u8)
            viz = cv.hconcat([frame, depth_color])

            # FPS “efectivo”
            t_now = time.perf_counter()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            if not args.no_display:
                cv.imshow("MiDaS Depth (video)", viz)
                # Cerrar si la ventana se cierra con 'X'
                if window_should_close("MiDaS Depth (video)"):
                    break
            if writer_video is not None:
                writer_video.write(viz)

            # Guardados periódicos
            if args.save_every_n > 0 and (idx % args.save_every_n == 0):
                base = Path(args.path).stem + f"_{idx:06d}"
                save_artifacts(depth, depth_u8, depth_color, out_dir, base, args)

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "video",
                idx,
                Path(args.path).name,
                args.model,
                device.type,
                args.size,
                int(args.fp16 and device.type == "cuda"),
                int(args.bilateral),
                frame.shape[1],
                frame.shape[0],
                f"{fps:.3f}",
            ])
            csv_file.flush()

            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break

            key = cv.waitKey(1) & 0xFF
            if key in (ord('q'), ord('x'), 27):
                break

    finally:
        cap.release()
        if writer_video is not None:
            writer_video.release()
        cv.destroyAllWindows()
        csv_file.close()


def process_images(args, midas, transform, device):
    img_dir = Path(args.path)
    if not img_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise RuntimeError("No se encontraron imágenes en la carpeta.")

    out_dir = get_output_dir(args)
    ensure_dir(out_dir)
    csv_file, writer = write_csv_header(out_dir / "log.csv")

    try:
        for idx, img_path in enumerate(files):
            frame = cv.imread(str(img_path), cv.IMREAD_COLOR)
            if frame is None:
                print(f"Aviso: no se pudo leer {img_path}")
                continue
            t0 = time.perf_counter()
            depth = run_inference(frame, midas, transform, device, args.size, args.fp16)
            t1 = time.perf_counter()
            fps = 1.0 / max(t1 - t0, 1e-6)

            depth_norm = percentile_normalize(depth, args.p_low, args.p_high)
            depth_u8 = (depth_norm * 255).astype(np.uint8)
            depth_u8 = apply_postprocess(depth_u8, args.bilateral)
            depth_color = colorize(depth_u8)

            # Mostrar y guardar
            if not args.no_display:
                viz = cv.hconcat([frame, depth_color])
                cv.imshow("MiDaS Depth (images)", viz)
                # Salir si el usuario cierra la ventana con 'X'
                if window_should_close("MiDaS Depth (images)"):
                    break
                # O con teclas 'q', 'x' o ESC
                key = cv.waitKey(1) & 0xFF
                if key in (ord('q'), ord('x'), 27):
                    break

            base = img_path.stem
            save_artifacts(depth, depth_u8, depth_color, out_dir, base, args)

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "images",
                idx,
                img_path.name,
                args.model,
                device.type,
                args.size,
                int(args.fp16 and device.type == "cuda"),
                int(args.bilateral),
                frame.shape[1],
                frame.shape[0],
                f"{fps:.3f}",
            ])
            csv_file.flush()

            if args.max_frames and idx + 1 >= args.max_frames:
                break
    finally:
        cv.destroyAllWindows()
        csv_file.close()


def save_artifacts(depth_f32, depth_u8, depth_color_bgr, out_dir: Path, base: str, args):
    # Profundidad cruda (float32) en .npy
    if args.save_raw:
        np.save(out_dir / f"{base}_depth.npy", depth_f32)

    # PNG 16-bit de la versión normalizada (escala 0..65535)
    if args.save_depth16:
        depth_norm = percentile_normalize(depth_f32, args.p_low, args.p_high)
        depth_u16 = (depth_norm * 65535.0).astype(np.uint16)
        cv.imwrite(str(out_dir / f"{base}_depth16.png"), depth_u16)

    # Visualización coloreada
    if args.save_color:
        cv.imwrite(str(out_dir / f"{base}_depth_color.png"), depth_color_bgr)


# -----------------------------
# Main / CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Demo híbrida de estimación de profundidad con MiDaS (webcam/video/imagenes) + PDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="DPT_Hybrid",
                    choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                    help="Variante de MiDaS a utilizar")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"],
                    help="Dispositivo de cómputo")
    ap.add_argument("--size", type=int, default=518, help="Tamaño objetivo del lado mayor")
    ap.add_argument("--fp16", action="store_true", help="Usar FP16 (solo CUDA)")

    ap.add_argument("--source", default="webcam",
                    choices=["webcam", "video", "images"],
                    help="Fuente de entrada")
    ap.add_argument("--path", type=str, default="", help="Ruta de video o carpeta de imágenes")
    ap.add_argument("--backend", type=str, default="auto",
                    choices=["auto", "avfoundation", "dshow", "v4l2"],
                    help="Backend de captura (webcam/video)")
    ap.add_argument("--cam-index", type=int, default=None, help="Índice de cámara (webcam)")
    ap.add_argument("--cam-width", type=int, default=640, help="Ancho de captura webcam")
    ap.add_argument("--cam-height", type=int, default=480, help="Alto de captura webcam")

    ap.add_argument("--bilateral", action="store_true", help="Aplicar filtro bilateral (PDI)")
    ap.add_argument("--p-low", type=float, default=2.0, help="Percentil bajo para normalización")
    ap.add_argument("--p-high", type=float, default=98.0, help="Percentil alto para normalización")
    ap.add_argument("--ema-alpha", type=float, default=0.85,
                    help="Coeficiente EMA para suavizado temporal (None desactiva)",
                    )
    ap.add_argument("--no-display", action="store_true", help="No mostrar ventanas (modo batch/headless)")

    ap.add_argument("--output-dir", type=str, default="outputs", help="Carpeta de salida")
    ap.add_argument("--output-subdir", type=str, default="", help="Nombre de subcarpeta dentro del directorio de salida (se creará si no existe).")
    ap.add_argument("--use-model-subdir", action="store_true", help="Guardar salidas dentro de una subcarpeta con el nombre del modelo (p. ej., outputs/DPT_Large).")
    ap.add_argument("--save-raw", action="store_true", help="Guardar profundidad cruda .npy (float32)")
    ap.add_argument("--save-depth16", action="store_true", help="Guardar PNG 16-bit (normalizada)")
    ap.add_argument("--save-color", action="store_true", help="Guardar PNG coloreado (visualización)")
    ap.add_argument("--save-video", action="store_true", help="Exportar video coloreado (para --source video)")
    ap.add_argument("--record-webcam", action="store_true", help="Grabar MP4 del stream de webcam (vista combinada)")
    ap.add_argument("--webcam-fps", type=float, default=25.0, help="FPS de salida para grabación de webcam")

    ap.add_argument("--save-every-n", type=int, default=0,
                    help="Guardar artefactos cada N frames (0=desactivar)")
    ap.add_argument("--max-frames", type=int, default=0, help="Procesar a lo más N frames (0=sin límite)")

    args = ap.parse_args()
    # Normaliza parámetros
    if args.ema_alpha is not None and args.ema_alpha <= 0:
        args.ema_alpha = None
    return args


def main():
    args = parse_args()

    device = pick_device(args.device)
    print(f"[INFO] Dispositivo: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Cargar modelo y transform correspondiente
    midas, transform = load_midas(args.model, device, args.fp16)

    # Rutas de salida
    ensure_dir(get_output_dir(args))

    # Despacho según fuente
    if args.source == "webcam":
        process_webcam(args, midas, transform, device)
    elif args.source == "video":
        if not args.path:
            print("Error: --path debe apuntar a un archivo de video.")
            sys.exit(2)
        process_video(args, midas, transform, device)
    elif args.source == "images":
        if not args.path:
            print("Error: --path debe apuntar a una carpeta con imágenes.")
            sys.exit(2)
        process_images(args, midas, transform, device)
    else:
        raise ValueError("Fuente no soportada")


if __name__ == "__main__":
    main()