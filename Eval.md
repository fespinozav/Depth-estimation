### Métricas de evaluación (qué significan)

- **AbsRel (error relativo absoluto)**: promedio de |p − g| / g. Más bajo es mejor.  
  `AbsRel = (1/N) * Σ |p_i − g_i| / g_i`
- **RMSE (raíz del error cuadrático medio)**: error en unidades físicas (m, cm). Sensible a outliers.  
  `RMSE = sqrt( (1/N) * Σ (p_i − g_i)^2 )`
- **δ a umbrales**: % de píxeles cuya razón `max(p/g, g/p)` < umbral.  
  Se reporta `δ<1.25`, `δ<1.25²` (~1.56) y `δ<1.25³` (~1.95). Más alto es mejor.

**Notas**  
- En modelos **no métricos** (MiDaS/DA-V2 relativa) alinear escala antes de evaluar (`--align median` o `--align ls`).  
- En **KITTI** suele usarse recorte **Eigen** (`--crop eigen`) y conversión de GT 16-bit con `--gt-16bit-scale 1/256`.  
- Solo se evalúan píxeles válidos del GT (no cero/inf) y dentro de los límites `--min-depth/--max-depth`.

**Ejemplo (CSV + resumen Markdown):**
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/demo_sala \
  --gt-dir   /ruta/gt_kitti \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align median --crop eigen --resize-pred \
  --out outputs/eval_kitti.csv \
  --md-out outputs/eval_kitti.md


# Evaluación de profundidad — Métricas y uso de `eval_depth.py`

Este documento explica las métricas empleadas y cómo ejecutar el script `eval_depth.py` para evaluar mapas de profundidad frente a su *ground truth*.

---

## Métricas de evaluación (qué significan)

- **AbsRel (error relativo absoluto)**: promedio de |p − g| / g. Más bajo es mejor.  
  `AbsRel = (1/N) * Σ |p_i − g_i| / g_i`
- **RMSE (raíz del error cuadrático medio)**: error en unidades físicas (m, cm). Sensible a *outliers*.  
  `RMSE = sqrt( (1/N) * Σ (p_i − g_i)^2 )`
- **δ a umbrales**: % de píxeles cuya razón `max(p/g, g/p)` < umbral.  
  Se reporta `δ<1.25`, `δ<1.25²` (~1.56) y `δ<1.25³` (~1.95). Más alto es mejor.

**Notas**  
- En modelos **no métricos** (MiDaS / DA‑V2 relativa) alinear escala antes de evaluar (`--align median` o `--align ls`).  
- En **KITTI** suele usarse recorte **Eigen** (`--crop eigen`) y conversión de GT 16‑bit con `--gt-16bit-scale 1/256` (≈ 0.00390625).  
- Solo se evalúan píxeles válidos del GT (no cero/inf) y dentro de los límites `--min-depth/--max-depth`.

---

## Uso de `eval_depth.py`

El script acepta predicciones en **`.npy` (float32)** o **PNG/TIFF 16‑bit**, y GT en los mismos formatos. También puede generar un **resumen Markdown** con medias y desviaciones estándar.

### 1) KITTI (GT 16‑bit, recorte Eigen, predicciones `.npy`)
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/demo_sala \
  --gt-dir   /ruta/gt_kitti \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align median --crop eigen --resize-pred \
  --out outputs/eval_kitti.csv \
  --md-out outputs/eval_kitti.md
```

### 2) NYU (GT `.npy`, predicciones **PNG 16‑bit** normalizadas)
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Large/testA \
  --gt-dir   /ruta/gt_nyu_npy \
  --pred-type png --gt-type npy \
  --pred-16bit-scale 1/65535 \
  --align ls \
  --out outputs/eval_nyu.csv \
  --md-out outputs/eval_nyu.md
```

### 3) Modelos con salida **métrica** (p. ej., ZoeDepth, DA‑V2 métrico)
```bash
python eval_depth.py \
  --pred-dir outputs/ZoeDepth/demo \
  --gt-dir   /ruta/gt_metrico \
  --pred-type npy --gt-type png \
  --align none \
  --out outputs/eval_metrico.csv
```

---

## Parámetros clave

- `--align {none, median, ls, scale}`: método de alineación de escala.
- `--pred-16bit-scale`, `--gt-16bit-scale`: factores para convertir 16‑bit a unidades de trabajo (p. ej., `1/256` en KITTI).
- `--crop eigen`: aplica recorte estándar de KITTI.
- `--resize-pred`: redimensiona la predicción al tamaño del GT si difieren.
- `--md-out`: produce un resumen en **Markdown** además del **CSV**.

---

## Resultados esperados (lectura rápida)

- **AbsRel** y **RMSE**: **más bajos** → mejor.  
- **δ<1.25**, **δ<1.25²**, **δ<1.25³**: **más altos** → mejor.  
- El **CSV** incluye resultados por imagen y un promedio al final. El **Markdown** (si se solicita) resume **media** y **desviación estándar** por métrica.

---

## Trazabilidad

- Las rutas de entrada (predicciones y GT) quedan registradas en el **CSV** y en el **Markdown**.  
- Se recomienda versionar el commit de pesos/modelo y los parámetros usados para asegurar **reproducibilidad**.