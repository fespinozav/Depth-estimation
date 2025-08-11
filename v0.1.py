import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt

# Selección automática de dispositivo (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando GPU CUDA")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Usando GPU MPS (Apple Silicon / macOS)")
else:
    device = torch.device("cpu")
    print("Usando CPU")

# Cargar modelo y transformaciones
model_type = "DPT_Large"  # También puedes usar "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Abrir cámara
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # más pequeño = más rápido
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se recibió frame. Saliendo...")
        break

    # Convertir a RGB para MiDaS
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Aplicar transformaciones
    input_batch = transform(img_rgb).to(device)

    # Inferencia
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalizar el mapa de profundidad
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)

    # Aplicar colormap para visualización
    depth_colored = cv.applyColorMap(depth_map, cv.COLORMAP_MAGMA)

    # Mostrar el resultado
    cv.imshow('Depth Estimation (MiDaS)', depth_colored)

    # Presiona 'q' para salir
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()