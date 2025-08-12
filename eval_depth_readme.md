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
  --pred-dir outputs/DPT_Hybrid/kitti_eval \
  --gt-dir   outputs/DPT_Hybrid/kitti_eval \
  --pred-type png --gt-type png \
  --pred-16bit-scale 1.5259e-5 --gt-16bit-scale 1.5259e-5 \
  --align none --resize-pred \
  --out outputs/DPT_Hybrid/self_eval.csv


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
# Evaluación de profundidad — Guía rápida, cambios recientes y resultados

Este documento resume **qué miden las métricas**, **cómo usar `eval_depth.py`** (con los cambios más recientes) y deja **resultados + conclusiones** para el set `val_selection_cropped` de KITTI.

---

## Novedades del evaluador (`eval_depth.py`)

- **Emparejamiento de archivos**
  - `--pairing name` *(por defecto)*: empareja por **basename** normalizado.
  - `--pairing sorted`: empareja **por orden lexicográfico** (ignora nombres).
  - `--allow-mismatch`: con `sorted`, si #pred ≠ #gt, evalúa hasta el mínimo común.
  - `--debug-pairs`: imprime los **primeros pares** emparejados (útil para ver si el orden es correcto).

- **Normalización automática de nombres**
  - En **pred**: se ignoran `*_depth_color.*` y se quitan sufijos `*_depth`, `*_depth16`.
  - En **GT**: se quitan prefijos `groundtruth_depth_` y `velodyne_raw_`.

- **Alineación mejorada**
  - `--align lss`: **least squares scale+shift** (ajusta `s` y `t` en `s·pred + t`).
  - También disponibles: `none`, `median`, `ls` (solo escala) y `scale` (factor fijo).

- **Resumen Markdown**
  - `--md-out`: genera un `.md` con **medias** y **desviaciones estándar** por métrica.

> Nota: en **`val_selection_cropped`** el GT ya viene recortado; **no** uses `--crop eigen` para ese set.

---

## Métricas de evaluación (qué significan)

- **AbsRel (error relativo absoluto)**: promedio de |p − g| / g. **Más bajo es mejor**.  
  `AbsRel = (1/N) * Σ |p_i − g_i| / g_i`
- **RMSE (raíz del error cuadrático medio)**: error en unidades físicas (m). Penaliza *outliers*. **Más bajo es mejor**.  
  `RMSE = sqrt( (1/N) * Σ (p_i − g_i)^2 )`
- **δ a umbrales**: % de píxeles con razón `max(p/g, g/p)` < umbral. **Más alto es mejor**.  
  Se reporta `δ<1.25`, `δ<1.25²` (~1.56) y `δ<1.25³` (~1.95).

**Notas**  
- Modelos **no métricos** (MiDaS) requieren **alineación** (`median`, `ls` o `lss`).  
- En KITTI 16‑bit, convierte a metros con **1/256** → `--gt-16bit-scale 0.00390625`.  
- Se evalúan solo píxeles válidos (no cero/inf) y dentro de `--min-depth/--max-depth`.

---

## Uso rápido

### 0) Prueba de vida (sin GT)
Compara la carpeta consigo misma (debería dar ≈0):
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_eval \
  --gt-dir   outputs/DPT_Hybrid/kitti_eval \
  --pred-type png --gt-type png \
  --pred-16bit-scale 1/65535 --gt-16bit-scale 1/65535 \
  --align none --resize-pred \
  --out outputs/DPT_Hybrid/self_eval.csv
```

### 1) KITTI — `val_selection_cropped` (rutas del proyecto)

**Generar predicciones** sobre las RGB:
```bash
python midas_depth_app.py \
  --source images \
  --path "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/rgb_images" \
  --model DPT_Hybrid \
  --use-model-subdir --output-subdir kitti_val \
  --save-raw --save-depth16 --no-display
```

**Evaluar (recomendado con `.npy`)**:
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align lss --resize-pred \
  --pairing sorted --debug-pairs \
  --out   outputs/DPT_Hybrid/kitti_val_eval_lss.csv \
  --md-out outputs/DPT_Hybrid/kitti_val_eval_lss.md
```

**Alternativa (predicciones PNG 16‑bit normalizadas)**:
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type png --gt-type png \
  --pred-16bit-scale 1/65535 \
  --gt-16bit-scale   0.00390625 \
  --align lss --resize-pred \
  --pairing sorted \
  --out   outputs/DPT_Hybrid/kitti_val_eval_png_lss.csv \
  --md-out outputs/DPT_Hybrid/kitti_val_eval_png_lss.md
```

---

## Resultados (corrida de referencia con DPT_Hybrid)

Comando utilizado:
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align median --resize-pred \
  --pairing sorted --debug-pairs \
  --out outputs/DPT_Hybrid/kitti_val_eval_sorted.csv
```

**Métricas reportadas:**

| #Imgs | Píxeles válidos | AbsRel | RMSE  | δ<1.25 | δ<1.25² | δ<1.25³ |
|:----:|-----------------:|-------:|------:|-------:|--------:|--------:|
| 16   | 1,744,738        | 0.5713 | 14.527 | 0.2147 | 0.4210  | 0.6179  |

**Archivo CSV**: `outputs/DPT_Hybrid/kitti_val_eval_sorted.csv`

> Nota: estas cifras usan `--align median` (solo escala). En modelos **no métricos** como MiDaS, suelen mejorar con `--align lss` (escala + shift). Se recomienda volver a correr con `lss` y comparar.

---

## Conclusiones y comentarios finales

- **MiDaS (DPT_Hybrid) produce profundidad relativa**, por lo que **la alineación** respecto a GT es clave.  
  - Con `median/ls` (solo escala) puede quedar un **sesgo aditivo** no corregido → **AbsRel/RMSE altos**.  
  - `lss` (escala+shift) suele **bajar AbsRel** y **subir δ** al corregir ese sesgo por escena.
- El set `val_selection_cropped` ya viene **recortado**; evitar `--crop eigen` aquí.  
- **Emparejamiento `sorted`** es práctico cuando los nombres no calzan; verifica el orden con `--debug-pairs`. Si #pred ≠ #gt, usa `--allow-mismatch`.
- **Mejoras sugeridas**:
  - Probar **`DPT_Large`** (mejor precisión que Hybrid) con el mismo pipeline.  
  - Evaluar un modelo **métrico** (p. ej., ZoeDepth o Depth Anything V2 métrico) con `--align none` para comparar escalas en metros.  
  - Añadir **SILog** como métrica complementaria (invariante a escala) si vas a comparar estrictamente modelos no métricos.
- **Buenas prácticas**:
  - Mantener fijos `--pairing`, `--align`, `--min/max-depth` al comparar entre modelos.  
  - Guardar **CSV** en cada corrida y anotar commit/pesos usados para **reproducibilidad**.

---

## Instructivo de uso (CLI) — paso a paso

Este instructivo resume los **comandos exactos de terminal (bash/zsh)** que hemos usado con éxito para generar predicciones y evaluar en KITTI `val_selection_cropped`.

> **Rutas usadas (ajústalas si cambian):**
> - RGB: `/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/rgb_images`
> - GT : `/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth`

### 1) Generar predicciones (MiDaS — DPT_Hybrid)

```bash
python midas_depth_app.py \
  --source images \
  --path \
  "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/rgb_images" \
  --model DPT_Hybrid \
  --use-model-subdir --output-subdir kitti_val \
  --save-raw --save-depth16 --no-display
```
Salidas esperadas (en `outputs/DPT_Hybrid/kitti_val/`):
- `*_depth.npy` (float32, recomendado para evaluar)
- `*_depth16.png` (16‑bit normalizado 0..65535, opcional)

### 2) Verificar que hay archivos
```bash
ls -1 outputs/DPT_Hybrid/kitti_val | head
ls -1 \
"/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
| head
```

### 3) Evaluación recomendada (pred `.npy` + alineación `lss`)

`lss` corrige **escala y sesgo** (scale+shift), ideal para modelos **no métricos** como MiDaS. Usamos `--pairing sorted` para ignorar nombres y emparejar por orden.

```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir \
  "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align lss --resize-pred \
  --pairing sorted --debug-pairs \
  --out   outputs/DPT_Hybrid/kitti_val_eval_lss.csv \
  --md-out outputs/DPT_Hybrid/kitti_val_eval_lss.md
```

### 4) Alternativa: evaluar con PNG 16‑bit normalizados

```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir \
  "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type png --gt-type png \
  --pred-16bit-scale 1/65535 \
  --gt-16bit-scale   0.00390625 \
  --align lss --resize-pred \
  --pairing sorted \
  --out   outputs/DPT_Hybrid/kitti_val_eval_png_lss.csv \
  --md-out outputs/DPT_Hybrid/kitti_val_eval_png_lss.md
```

### 5) Si los nombres no calzan: generar lista de intersección

Usa esta receta para construir `names.txt` con **basenames comunes**, quitando sufijos de pred (`_depth`, `_depth16`) y el prefijo de GT (`groundtruth_depth_`):

```bash
ls outputs/DPT_Hybrid/kitti_val \
| sed -E 's/(_depth16|_depth)?\.(png|tif|tiff|npy)$//' \
| sort -u > /tmp/pred.txt

ls \
"/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
| sed -E 's/\.(png|tif|tiff)$//' \
| sed -E 's/^groundtruth_depth_//' \
| sort -u > /tmp/gt.txt

comm -12 /tmp/pred.txt /tmp/gt.txt > names.txt
wc -l names.txt
head names.txt

# Evalúa solo esos nombres
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir \
  "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align lss --resize-pred \
  --list names.txt \
  --out outputs/DPT_Hybrid/kitti_val_eval_lss_list.csv
```

### 6) Prueba de vida (auto‑GT):

Compara la carpeta consigo misma (debe dar ≈0):
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir   outputs/DPT_Hybrid/kitti_val \
  --pred-type png --gt-type png \
  --pred-16bit-scale 1/65535 --gt-16bit-scale 1/65535 \
  --align none --resize-pred \
  --out outputs/DPT_Hybrid/self_eval.csv
```

### 7) Tips y errores comunes

- **zsh y comentarios**: en zsh el `#` no comenta al final de línea a menos que actives:
  ```bash
  setopt interactivecomments
  ```
- **“No se encontraron pares pred/gt”**: prueba `--pairing sorted --debug-pairs` o usa el flujo de `names.txt` (Paso 5).
- **“No existe carpeta …”**: revisa la ruta; usa **rutas absolutas** para evitar confusión.
- **Desbalance de archivos**: con `--pairing sorted` añade `--allow-mismatch` para evaluar hasta el mínimo común.
- **Clipping**: puedes limitar el rango útil con `--min-depth 0.1 --max-depth 80`.

### 8) Variante con DPT_Large

```bash
# Generar predicciones
python midas_depth_app.py \
  --source images \
  --path \
  "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/rgb_images" \
  --model DPT_Large \
  --use-model-subdir --output-subdir kitti_val_large \
  --save-raw --no-display

# Evaluar (npy + lss)
python eval_depth.py \
  --pred-dir outputs/DPT_Large/kitti_val_large \
  --gt-dir \
  "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align lss --resize-pred --pairing sorted \
  --out outputs/DPT_Large/kitti_val_large_eval_lss.csv
```