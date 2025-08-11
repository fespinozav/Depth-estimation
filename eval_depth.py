#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de mapas de profundidad: AbsRel, RMSE y δ<1.25 / δ<1.25^2 / δ<1.25^3.

Soporta predicciones en .npy (float32) o PNG/TIFF 16-bit y ground-truth en .npy o 16-bit.
Incluye opciones de alineación de escala (median, least-squares) para modelos de profundidad relativa (MiDaS, DA-V2).
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2 as cv

VALID_IMG_EXTS = {".png", ".tif", ".tiff"}
VALID_NPY_EXTS = {".npy"}

# ---------- Carga ----------
def load_depth(path: Path, dtype: str, bit16_scale: float) -> np.ndarray:
    """
    Carga un mapa de profundidad desde .npy o imagen 16-bit. Devuelve float32.
    """
    ext = path.suffix.lower()
    if dtype == "auto":
        if ext in VALID_NPY_EXTS:
            dtype = "npy"
        elif ext in VALID_IMG_EXTS:
            dtype = "png"
        else:
            raise ValueError(f"Extensión no soportada para auto: {path}")

    if dtype == "npy":
        arr = np.load(str(path)).astype(np.float32)
    elif dtype == "png":
        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"No se puede leer {path}")
        if img.ndim == 3:  # por si viniera con canales
            img = img[..., 0]
        if img.dtype == np.uint16:
            arr = img.astype(np.float32) * float(bit16_scale)
        elif img.dtype == np.uint8:
            arr = img.astype(np.float32) * float(bit16_scale)
        else:
            arr = img.astype(np.float32)
    else:
        raise ValueError(f"dtype no reconocido: {dtype}")
    return arr

def resize_to(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if arr.shape[:2] == (h, w):
        return arr
    return cv.resize(arr, (w, h), interpolation=cv.INTER_LINEAR).astype(np.float32)

# ---------- Alineación ----------
def align_prediction(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, method: str, scale_value: float = 1.0):
    """
    Ajusta la escala de 'pred' para compararla con 'gt'.
    method: none | median | ls | scale
    """
    eps = 1e-8
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if not np.any(m):
        return pred, 1.0

    if method == "none":
        return pred, 1.0
    if method == "median":
        med_p = np.median(pred[m]); med_g = np.median(gt[m])
        s = (med_g + eps) / (med_p + eps)
        return pred * s, s
    if method == "ls":
        p = pred[m]; g = gt[m]
        s = float(np.sum(p * g) / (np.sum(p * p) + eps))
        return pred * s, s
    if method == "scale":
        return pred * float(scale_value), float(scale_value)
    raise ValueError(f"Método de alineación no soportado: {method}")

# ---------- Métricas ----------
def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if not np.any(m):
        return {k: float("nan") for k in ["absrel", "rmse", "delta1", "delta2", "delta3", "n_valid"]}

    p = pred[m]; g = gt[m]
    absrel = float(np.mean(np.abs(g - p) / (g + eps)))
    rmse = float(np.sqrt(np.mean((g - p) ** 2)))

    ratio = np.maximum(g / (p + eps), p / (g + eps))
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    return {"absrel": absrel, "rmse": rmse, "delta1": delta1, "delta2": delta2, "delta3": delta3, "n_valid": int(m.sum())}

# ---------- Utilidades ----------
def eigen_crop_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
    x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
    mask[y1:y2, x1:x2] = True
    return mask

def make_mask(gt: np.ndarray, min_depth: float, max_depth: float, crop: str) -> np.ndarray:
    h, w = gt.shape[:2]
    m = np.isfinite(gt)
    if min_depth is not None:
        m &= gt > float(min_depth)
    if max_depth is not None:
        m &= gt < float(max_depth)
    if crop == "eigen":
        m &= eigen_crop_mask(h, w)
    return m

def intersect_basenames(pred_dir: Path, gt_dir: Path, pred_type: str, gt_type: str, file_list: Path = None) -> List[str]:
    def collect(dirp: Path, dtype: str):
        mapping = {}
        if dtype in ("auto", "png"):
            for ext in VALID_IMG_EXTS:
                for p in dirp.glob(f"*{ext}"):
                    mapping[p.stem] = p
        if dtype in ("auto", "npy"):
            for p in dirp.glob("*.npy"):
                mapping[p.stem] = p
        return mapping
    pred_map = collect(pred_dir, pred_type)
    gt_map = collect(gt_dir, gt_type)
    if file_list is not None and file_list.exists():
        names = []
        for line in file_list.read_text().splitlines():
            n = line.strip()
            if n and n in pred_map and n in gt_map:
                names.append(n)
        return names
    return sorted(set(pred_map.keys()) & set(gt_map.keys()))

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluación de profundidad: AbsRel, RMSE, δ<1.25/1.25^2/1.25^3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pred-dir", required=True, type=Path, help="Carpeta con predicciones (.npy / 16-bit)")
    ap.add_argument("--gt-dir", required=True, type=Path, help="Carpeta con ground-truth (.npy / 16-bit)")
    ap.add_argument("--pred-type", default="auto", choices=["auto", "npy", "png"], help="Tipo de archivo de predicción")
    ap.add_argument("--gt-type", default="auto", choices=["auto", "npy", "png"], help="Tipo de archivo de GT")
    ap.add_argument("--pred-16bit-scale", type=float, default=1.0/65535.0, help="Escala para convertir pred 16-bit a float")
    ap.add_argument("--gt-16bit-scale", type=float, default=1.0, help="Escala para convertir GT 16-bit (p. ej., 1/256 en KITTI)")
    ap.add_argument("--align", default="median", choices=["none", "median", "ls", "scale"], help="Alineación de escala para la predicción")
    ap.add_argument("--scale", type=float, default=1.0, help="Escala fija cuando --align scale")
    ap.add_argument("--min-depth", type=float, default=1e-3, help="Mínima profundidad válida")
    ap.add_argument("--max-depth", type=float, default=80.0, help="Máxima profundidad válida")
    ap.add_argument("--crop", default="none", choices=["none", "eigen"], help="Recorte del área evaluada (KITTI usa 'eigen')")
    ap.add_argument("--list", type=Path, default=None, help="Archivo con lista de nombres (sin extensión) a evaluar")
    ap.add_argument("--resize-pred", action="store_true", help="Si formas no coinciden, redimensiona pred al tamaño del GT")
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_results.csv"), help="Ruta del CSV de salida")
    return ap.parse_args()

def main():
    args = parse_args()
    pred_dir: Path = args.pred_dir
    gt_dir: Path = args.gt_dir

    args.out.parent.mkdir(parents=True, exist_ok=True)

    names = intersect_basenames(pred_dir, gt_dir, args.pred_type, args.gt_type, args.list)
    if not names:
        raise SystemExit("No se encontraron pares pred/gt con nombres coincidentes.")

    rows = []
    agg = {"absrel": [], "rmse": [], "delta1": [], "delta2": [], "delta3": [], "n_valid": []}

    for name in names:
        # Encuentra archivos
        def find_file(dirp: Path, dtype: str) -> Path:
            if dtype == "npy":
                p = dirp / f"{name}.npy"
                if p.exists(): return p
            if dtype == "png":
                for ext in VALID_IMG_EXTS:
                    p = dirp / f"{name}{ext}"
                    if p.exists(): return p
            if dtype == "auto":
                p = dirp / f"{name}.npy"
                if p.exists(): return p
                for ext in VALID_IMG_EXTS:
                    p = dirp / f"{name}{ext}"
                    if p.exists(): return p
            raise FileNotFoundError(f"No se encontró archivo para '{name}' en {dirp}")

        pred_path = find_file(pred_dir, args.pred_type)
        gt_path = find_file(gt_dir, args.gt_type)

        pred = load_depth(pred_path, "auto" if args.pred_type == "auto" else args.pred_type, args.pred_16bit_scale)
        gt   = load_depth(gt_path,   "auto" if args.gt_type   == "auto" else args.gt_type,   args.gt_16bit_scale)

        # Asegurar misma forma
        if pred.shape != gt.shape:
            if args.resize_pred:
                pred = resize_to(pred, gt.shape[:2])
            else:
                h = min(pred.shape[0], gt.shape[0]); w = min(pred.shape[1], gt.shape[1])
                pred = pred[:h, :w]; gt = gt[:h, :w]

        mask = make_mask(gt, args.min_depth, args.max_depth, args.crop)
        pred_aligned, s = align_prediction(pred, gt, mask, args.align, args.scale)
        metrics = compute_metrics(pred_aligned, gt, mask)
        metrics["scale_used"] = s
        metrics["name"] = name
        rows.append(metrics)

        for k in ["absrel", "rmse", "delta1", "delta2", "delta3"]:
            if not np.isnan(metrics[k]): agg[k].append(metrics[k])
        agg["n_valid"].append(metrics["n_valid"])

    mean_metrics = {k: (float(np.mean(v)) if len(v) > 0 else float("nan")) for k, v in agg.items() if k != "n_valid"}
    mean_metrics["n_images"] = len(rows)
    mean_metrics["n_pixels"] = int(np.sum(agg["n_valid"])) if agg["n_valid"] else 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "absrel", "rmse", "delta1", "delta2", "delta3", "scale_used", "n_valid"])
        for r in rows:
            writer.writerow([r["name"], f"{r['absrel']:.6f}", f"{r['rmse']:.6f}", f"{r['delta1']:.6f}", f"{r['delta2']:.6f}", f"{r['delta3']:.6f}", f"{r['scale_used']:.6f}", r["n_valid"]])
        writer.writerow([])
        writer.writerow(["__mean__", f"{mean_metrics['absrel']:.6f}", f"{mean_metrics['rmse']:.6f}", f"{mean_metrics['delta1']:.6f}", f"{mean_metrics['delta2']:.6f}", f"{mean_metrics['delta3']:.6f}", "-", mean_metrics["n_images"]])

    print("=== Resultados ===")
    print(f"Imágenes: {mean_metrics['n_images']}  |  Píxeles válidos: {mean_metrics['n_pixels']}")
    print(f"AbsRel: {mean_metrics['absrel']:.6f}")
    print(f"RMSE  : {mean_metrics['rmse']:.6f}")
    print(f"δ<1.25 : {mean_metrics['delta1']:.6f}")
    print(f"δ<1.25²: {mean_metrics['delta2']:.6f}")
    print(f"δ<1.25³: {mean_metrics['delta3']:.6f}")
    print(f"CSV guardado en: {args.out}")

if __name__ == "__main__":
    main()