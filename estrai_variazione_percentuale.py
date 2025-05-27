import cv2
import numpy as np
import pytesseract
import re
from openvino.inference_engine import IECore

# Percorsi
model_xml = "models/intel/text-detection-0004/FP16/text-detection-0004.xml"
model_bin = "models/intel/text-detection-0004/FP16/text-detection-0004.bin"
image_path = "b613b680-126c-4be0-bc58-33718a551535.png"
output_txt = "variazioni_percentuali.txt"

# Se tesseract non è nel PATH, specifica qui:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Inizializza Inference Engine (MYRIAD)
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="MYRIAD")

# Prepara immagine
image = cv2.imread(image_path)
orig = image.copy()
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape
resized = cv2.resize(image, (w, h))
input_image = resized.transpose((2, 0, 1))[np.newaxis, :]

# Inference
result = exec_net.infer({input_blob: input_image})
boxes = result[output_blob][0][0]

# Estrai valori % con OCR
percentuali_estratte = []
for box in boxes:
    _, _, conf, x_min, y_min, x_max, y_max = box
    if conf > 0.5:
        xmin = int(x_min * orig.shape[1])
        ymin = int(y_min * orig.shape[0])
        xmax = int(x_max * orig.shape[1])
        ymax = int(y_max * orig.shape[0])
        roi = orig[ymin:ymax, xmin:xmax]
        testo = pytesseract.image_to_string(roi, config="--psm 6 -c tessedit_char_whitelist=0123456789+-,.%").strip()
        # Filtra solo stringhe contenenti %
        match = re.findall(r"[+-]?\d{1,3},\d{2}%", testo)
        percentuali_estratte.extend(match)

# Salva in file
with open(output_txt, "w", encoding="utf-8") as f:
    for percentuale in percentuali_estratte:
        f.write(percentuale + "\n")

print(f"✅ {len(percentuali_estratte)} variazioni salvate in: {output_txt}")
