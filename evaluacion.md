# Evaluación de profundidad — Guía práctica, métricas y resultados

Este documento explica **cómo evaluar mapas de profundidad** con `eval_depth.py`, qué significan las **métricas**, qué **cambios** incorpora el evaluador y deja **resultados de referencia** sobre `KITTI/val_selection_cropped`, con interpretación y recomendaciones. Se eliminaron duplicados y se mejoró la redacción para que sea directo y accionable.

---

## Novedades del evaluador (`eval_depth.py`)

- **Emparejamiento de archivos**
  - `--pairing name` *(por defecto)*: empareja por **basename** normalizado.
  - `--pairing sorted`: empareja **por orden lexicográfico**, ignorando nombres.
  - `--allow-mismatch`: con `sorted`, si #pred ≠ #gt, evalúa hasta el mínimo común.
  - `--debug-pairs`: imprime los **primeros pares** para verificar el emparejamiento.

- **Normalización automática de nombres**
  - En **pred**: ignora `*_depth_color.*` y quita sufijos `*_depth`, `*_depth16`.
  - En **GT**: quita prefijos `groundtruth_depth_` y `velodyne_raw_`.

- **Alineación mejorada**
  - `--align lss`: **least squares scale+shift** (ajusta escala y sesgo: `s·pred + t`).
  - También: `none`, `median`, `ls` (solo escala), `scale` (factor fijo).

- **Resumen Markdown**
  - `--md-out`: genera un `.md` con **medias** y **desviaciones estándar** por métrica.

> Nota: en `val_selection_cropped` el GT ya viene recortado; **no** uses `--crop eigen` aquí.

---

## Métricas (qué miden)

- **AbsRel (error relativo absoluto)** — Promedio de |p−g|/g. **Más bajo es mejor.**  
  `AbsRel = (1/N) * Σ |p_i − g_i| / g_i`
- **RMSE (raíz del error cuadrático medio)** — Error en **metros**; penaliza *outliers*. **Más bajo es mejor.**  
  `RMSE = sqrt( (1/N) * Σ (p_i − g_i)^2 )`
- **δ a umbrales** — % de píxeles con `max(p/g, g/p)` < umbral. **Más alto es mejor.**  
  Se reporta `δ<1.25`, `δ<1.25²` (~1.56) y `δ<1.25³` (~1.95).

**Detalles prácticos**
- Modelos **no métricos** (MiDaS) requieren **alineación** (`median`, `ls` o `lss`).  
- KITTI 16‑bit → metros con **1/256** (`--gt-16bit-scale 0.00390625`).  
- Se evalúan solo píxeles válidos del GT (no cero/inf) y dentro de `--min-depth/--max-depth`.

---

## Resultados de referencia (KITTI `val_selection_cropped`, DPT_Hybrid)

**Comando ejecutado** (emparejamiento por orden + alineación *median*):
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

**Métricas obtenidas**

| #Imgs | Píxeles válidos | AbsRel | RMSE  | δ<1.25 | δ<1.25² | δ<1.25³ |
|:----:|-----------------:|-------:|------:|-------:|--------:|--------:|
| 16   | 1,744,738        | 0.5713 | 14.527 | 0.2147 | 0.4210  | 0.6179  |

**Interpretación**
- **AbsRel 0.57 / δ<1.25 = 0.21**: el error relativo es alto y solo ~21% de los píxeles cae dentro de ±25% del GT.  
  Esto es **esperable** con modelos **no métricos** evaluados con una **única** reescala (median/ls): queda un **sesgo aditivo** no corregido.
- **RMSE 14.53 m**: valores de metros grandes reflejan tanto el **offset** como outliers a larga distancia.

**Acción recomendada**  
Repetir la evaluación con **`--align lss`** (escala **y** shift):
```bash
python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align lss --resize-pred \
  --pairing sorted \
  --out outputs/DPT_Hybrid/kitti_val_eval_lss.csv \
  --md-out outputs/DPT_Hybrid/kitti_val_eval_lss.md
```
En la práctica, **`lss` reduce AbsRel y sube δ** al remover el sesgo por escena. (Los números exactos dependen del conjunto y del modelo.)

**Próximos experimentos**
- Usar **`DPT_Large`** (mejora típica sobre Hybrid).  
- Probar un modelo **métrico** (ZoeDepth, Depth Anything V2 métrico) con `--align none` para comparar errores en **metros reales**.  
- Añadir **SILog** (RMSE invariante a escala) si se comparan solo modelos no métricos.

---

## Instructivo de uso (CLI)

> **Rutas del proyecto** (ajústalas si cambian):
> - RGB: `/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/rgb_images`  
> - GT : `/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth`

### 1) Generar predicciones (MiDaS — DPT_Hybrid)
```bash
python midas_depth_app.py \
  --source images \
  --path "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/rgb_images" \
  --model DPT_Hybrid \
  --use-model-subdir --output-subdir kitti_val \
  --save-raw --save-depth16 --no-display
```
Salidas (en `outputs/DPT_Hybrid/kitti_val/`):  
`*_depth.npy` (recomendado para evaluar) y `*_depth16.png` (16‑bit normalizado, opcional).

### 2) Verificar archivos
```bash
ls -1 outputs/DPT_Hybrid/kitti_val | head
ls -1 "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" | head
```

### 3) Evaluación recomendada (pred `.npy` + `lss`)
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

### 4) Alternativa: evaluar con PNG 16‑bit normalizados
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

### 5) Si los nombres no calzan: generar `names.txt`
```bash
ls outputs/DPT_Hybrid/kitti_val \
| sed -E 's/(_depth16|_depth)?\.(png|tif|tiff|npy)$//' \
| sort -u > /tmp/pred.txt

ls "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
| sed -E 's/\.(png|tif|tiff)$//' \
| sed -E 's/^groundtruth_depth_//' \
| sort -u > /tmp/gt.txt

comm -12 /tmp/pred.txt /tmp/gt.txt > names.txt
wc -l names.txt
head names.txt

python eval_depth.py \
  --pred-dir outputs/DPT_Hybrid/kitti_val \
  --gt-dir "/Users/felipeespinoza/Documents/GitHub/Depth-estimation/data/KITTI/val_selection_cropped/groundtruth_depth" \
  --pred-type npy --gt-type png \
  --gt-16bit-scale 0.00390625 \
  --align lss --resize-pred \
  --list names.txt \
  --out outputs/DPT_Hybrid/kitti_val_eval_lss_list.csv
```

### 6) Prueba de vida (auto‑GT)
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
- En **zsh**, el `#` no comenta al final de línea a menos que ejecutes: `setopt interactivecomments`.  
- “**No se encontraron pares pred/gt**”: usa `--pairing sorted --debug-pairs` o el flujo `names.txt`.  
- “**No existe carpeta …**”: usa **rutas absolutas**.  
- **Desbalance de archivos**: con `sorted`, agrega `--allow-mismatch` para evaluar hasta el mínimo común.  
- **Clipping**: limita rango con `--min-depth 0.1 --max-depth 80` si hay outliers.

---

## Conclusiones

- En monocular **no métrico** (MiDaS), la diferencia entre `median/ls` (solo escala) y **`lss`** (escala+shift) es **crítica**: `lss` corrige el sesgo por escena y **mejora AbsRel/δ**.  
- Mantén constantes `--pairing`, `--align` y los límites de profundidad al comparar modelos.  
- Para comparar **en metros**, incorpora un modelo **métrico** (ZoeDepth / DA‑V2 métrico) con `--align none`.  
- Versiona **CSV/MD** y parámetros usados para asegurar **reproducibilidad** en el informe final.