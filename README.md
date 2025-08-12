# Estimación de Profundidad Monocular (MiDaS) — App híbrida PDI + DL

Aplicación CLI para **estimar profundidad monocular** en **tiempo real** y por **lotes**, basada en **MiDaS (Intel‑ISL)** e integrada con técnicas de **Procesamiento Digital de Imágenes (PDI)**: normalización robusta por percentiles, filtro bilateral (preserva bordes) y **suavizado temporal EMA** para video.  
Funciona con **webcam**, **videos MP4** o **carpetas de imágenes**, y autodetecta **CUDA/MPS/CPU**. Incluye salida a **MP4** y artefactos reproducibles (PNG 16‑bit, .npy y log CSV).

<p align="center">
  <img src="midas_depth.gif" alt="Webcam test" width="600">
</p>

---

## ✨ Características
- **Entradas**: `webcam`, `video`, `images` (carpeta).  
- **Modelos MiDaS**: `DPT_Large`, `DPT_Hybrid`, `MiDaS_small` (aplica la *transform* adecuada por modelo).  
- **Hardware**: `--device auto|cuda|mps|cpu` + **FP16** opcional en CUDA.  
- **PDI**: normalización por percentiles (p2–p98), **bilateral** opcional, **EMA** para reducir *flicker* en video.  
- **Visualización**: RGB | Depth (colormap **MAGMA**) + **FPS**.  
- **Salidas**: `.npy` (float32 crudo), **PNG 16‑bit** normalizado, PNG coloreado y **MP4** (para `video` y `webcam` con `--record-webcam`).  
- **Grabación webcam**: `--record-webcam` graba MP4 del stream de webcam (FPS configurable con `--webcam-fps`).  
- **Organización de salidas**: `--use-model-subdir` crea `outputs/<MODELO>/`; `--output-subdir` añade `outputs/.../<SUBDIR>/`.
- **Usabilidad**: salir con teclas **q/x/ESC** o cerrando la ventana con **“X”**.  
- **Logs**: `outputs/log.csv` con parámetros y FPS por cuadro.

---

## 🧱 Requisitos
- Python 3.9+
- Dependencias mínimas:
  ```bash
  pip install torch opencv-python numpy
  ```
> Para GPU CUDA/MPS instala PyTorch acorde a tu plataforma (CUDA/cuDNN o Apple Silicon).

---

## 🚀 Uso rápido
El script principal es **`midas_depth_app.py`**.

### Webcam (demo en vivo)
```bash
python midas_depth_app.py --source webcam --backend avfoundation \
  --model DPT_Hybrid --bilateral --save-every-n 60 --save-color --save-depth16
```
- **Cerrar**: **q**, **x**, **ESC** o botón **“X”**.

### Webcam (grabando MP4)
```bash
python midas_depth_app.py --source webcam --backend avfoundation \
  --model DPT_Hybrid --record-webcam --webcam-fps 25 \
  --bilateral --save-every-n 60 --save-color --save-depth16
# genera: outputs/webcam_depth_colored.mp4 (o dentro de las subcarpetas configuradas)
```

### Video → exportar MP4 con profundidad
```bash
python midas_depth_app.py --source video --path ./samples/entrada.mp4 \
  --save-video --save-every-n 30 --save-color --save-depth16
# genera: outputs/depth_colored.mp4
```

### Carpeta de imágenes (batch, sin ventanas)
```bash
python midas_depth_app.py --source images --path ./images \
  --no-display --save-raw --save-depth16 --save-color
```

**Opcionales útiles**
```bash
# Forzar CUDA + FP16 y mayor tamaño de entrada
python midas_depth_app.py --device cuda --fp16 --size 640 --source video \
  --path ./samples/entrada.mp4 --save-video

# Guardar dentro de una subcarpeta por modelo (p. ej., outputs/DPT_Large/...)
python midas_depth_app.py --source images --path ./images \
  --model DPT_Large --use-model-subdir --save-color --save-depth16

# Guardar dentro de una subcarpeta personalizada (p. ej., outputs/kitti_eval/...)
python midas_depth_app.py --source video --path ./samples/entrada.mp4 \
  --save-video --output-subdir kitti_eval
```

---

## ⚙️ Parámetros (CLI)
| Flag | Default | Descripción |
|---|---|---|
| `--model` | `DPT_Hybrid` | `DPT_Large`, `DPT_Hybrid`, `MiDaS_small` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--size` | `518` | Lado mayor objetivo (con *letterbox*) |
| `--fp16` | `False` | FP16 (solo CUDA) |
| `--source` | `webcam` | `webcam`, `video`, `images` |
| `--path` | `""` | Ruta a video o carpeta de imágenes |
| `--backend` | `auto` | `avfoundation` (mac), `dshow` (win), `v4l2` (linux) |
| `--cam-index` | `None` | Índice de cámara |
| `--cam-width` | `640` | Ancho captura webcam |
| `--cam-height` | `480` | Alto captura webcam |
| `--bilateral` | `False` | Filtro bilateral (PDI) |
| `--p-low` | `2.0` | Percentil bajo para normalización |
| `--p-high` | `98.0` | Percentil alto para normalización |
| `--ema-alpha` | `0.85` | EMA temporal (video). `<=0` desactiva |
| `--no-display` | `False` | Modo *headless* (sin ventanas) |
| `--output-dir` | `outputs` | Carpeta de salidas |
| `--output-subdir` | `""` | Subcarpeta dentro de `--output-dir` |
| `--use-model-subdir` | `False` | Crear subcarpeta con el nombre del modelo (p. ej., `outputs/DPT_Large`) |
| `--save-raw` | `False` | Guarda `.npy` (float32) |
| `--save-depth16` | `False` | Guarda PNG 16‑bit normalizado |
| `--save-color` | `False` | Guarda PNG coloreado |
| `--save-video` | `False` | Exporta MP4 (solo `--source video`) |
| `--record-webcam` | `False` | Grabar MP4 del stream de webcam (vista combinada) |
| `--webcam-fps` | `25.0` | FPS de salida para grabación de webcam |
| `--save-every-n` | `0` | Guardar artefactos cada N frames |
| `--max-frames` | `0` | Límite de cuadros a procesar |

---

## 🧪 Pipeline y arquitectura
1. **Lectura** (webcam/video/imagen)  
2. **Prepro**: *letterbox* al tamaño objetivo  
3. **Inferencia MiDaS** (usa `dpt_transform` o `small_transform` según modelo)  
4. **Reescalado** del mapa a la resolución original  
5. **Normalización robusta** (percentiles p2–p98)  
6. **PDI**: filtro **bilateral** (opcional)  
7. **EMA temporal** (video)  
8. **Colorización** (MAGMA) y **visualización** con **FPS**  
9. **Guardado**: `.npy`, **PNG 16‑bit**, PNG coloreado y **MP4** (video)  
10. **Log CSV** con metadata de ejecución

---

## 📂 Estructura mínima
```
.
├── midas_depth_app.py
├── outputs/
│   ├── depth_colored.mp4
│   ├── *_depth16.png
│   ├── *_depth_color.png
│   ├── *_depth.npy
│   └── log.csv
└── samples/ (opcional)
```

**Con subcarpetas opcionales**
```
outputs/
└── DPT_Large/              # --use-model-subdir
    └── demo_sala/          # --output-subdir demo_sala
        ├── depth_colored.mp4
        ├── *_depth16.png
        ├── *_depth_color.png
        ├── *_depth.npy
        └── log.csv
```

---

## 📈 Evaluación
Si se dispone de ground truth (p. ej. **KITTI**), añade un `eval_depth.py` para calcular **AbsRel**, **RMSE**, **log10**, **δ<1.25/1.25²/1.25³**.  
Si no hay GT, entrega comparativas visuales (mosaicos input|depth, PNG 16‑bit) y **FPS**.

---

## 🧩 Próximos desarrollos
- Soporte adicional: **Depth Anything V2** (relativa/métrica), **ZoeDepth** (métrica).  
- Métricas cualitativas de bordes (iBims‑1‑like).

---

## 🙏 Créditos
- **MiDaS / DPT** — Intel Intelligent Systems Lab (ISL).  
- Este proyecto integra PDI + DL y usa pesos/modelos distribuidos por sus autores.

---

## © Licencia

- MIT License

---

## 🐞 Troubleshooting
- **No abre la webcam**: prueba `--backend avfoundation` (mac), `--backend dshow` (Windows) o `--backend v4l2` (Linux) y revisa `--cam-index`.  
- **Bajo FPS**: usa `--model MiDaS_small` o `DPT_Hybrid`, reduce `--size`, habilita `--fp16` en CUDA.  
- **Parpadeo en profundidad**: deja activado `--ema-alpha 0.85` y usa `--bilateral`.  
- **Ventana no cierra**: usa `q/x/ESC` o el botón **“X”**.

## 🔗 Referencias
- https://github.com/isl-org/MiDaS
- https://learnopencv.com/depth-anything/
- https://medium.com/@patriciogv/the-state-of-the-art-of-depth-estimation-from-single-images-9e245d51a315
- https://github.com/DepthAnything/Depth-Anything-V2
- https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
- ...
