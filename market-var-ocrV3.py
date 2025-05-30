import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openvino.runtime import Core
from scipy.ndimage import label, find_objects
import pytesseract

# ---------------------------
# Postprocessing for detection output
# ---------------------------
def decode_text_boxes(output_blob, image_shape, threshold=0.3):
    orig_h, orig_w = image_shape[:2]
    bin_map = output_blob[0, :, :, 0] > threshold

    labeled, num_features = label(bin_map.astype(np.uint8))
    slices = find_objects(labeled)

    boxes = []
    for sl in slices:
        y_min, y_max = sl[0].start, sl[0].stop
        x_min, x_max = sl[1].start, sl[1].stop

        x_scale = orig_w / bin_map.shape[1]
        y_scale = orig_h / bin_map.shape[0]

        x_min = int(x_min * x_scale)
        x_max = int(x_max * x_scale)
        y_min = int(y_min * y_scale)
        y_max = int(y_max * y_scale)

        boxes.append(((x_min, y_min), (x_max, y_max)))
    return boxes

# ---------------------------
# OCR pipeline (Hybrid: OpenVINO Detection + Tesseract OCR)
# ---------------------------
def run_ocr_with_boxes(image_path):
    core = Core()
    det_model_path = "/home/italygourmet_co/openvino-ocr-env/intel/text-detection-0004/FP32/text-detection-0004.xml"

    try:
        det_model = core.compile_model(det_model_path, "CPU")
    except Exception as e:
        print(f"Errore nel caricamento del modello di detection OpenVINO: {e}")
        exit()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Errore: Impossibile caricare l'immagine da {image_path}")
        exit()

    orig = image.copy()
    h, w = image.shape[:2]

    input_shape = det_model.input(0).shape
    _, H, W, _ = input_shape
    resized = cv2.resize(image, (W, H))
    blob = np.expand_dims(resized, axis=0).astype(np.float32)

    output_tensor = det_model([blob])
    output = list(output_tensor.values())[0]

    boxes = decode_text_boxes(output, image.shape)
    results = []

    for (x_min, y_min), (x_max, y_max) in boxes:
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        config = "--psm 6"
        text = pytesseract.image_to_string(roi, config=config).strip()
        results.append((y_min, x_min, text, (x_min, y_min, x_max, y_max)))

    results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
    return results_sorted, orig

# ---------------------------
# Estrai indice e variazione % da tutte le righe (inclusa intestazione)
# ---------------------------
def estrai_indici_variazioni(ocr_data):
    y_clusters = []
    y_tolerance = 10
    for item in ocr_data:
        y = item[0]
        matched = False
        for cluster in y_clusters:
            if abs(cluster[0] - y) <= y_tolerance:
                cluster[1].append(item)
                matched = True
                break
        if not matched:
            y_clusters.append((y, [item]))

    dati = []
    for _, cluster_items in y_clusters:
        cluster_items.sort(key=lambda d: d[1])
        texts = [d[2] for d in cluster_items]
        riga = " ".join(texts)

        var_percent = next((t for t in texts if '%' in t), None)
        indice = texts[0] if len(texts) > 0 else ""

        if indice and var_percent and not (indice.lower() == "indice" and var_percent.lower() == "variazione %"):
            dati.append((indice, var_percent))

    df = pd.DataFrame(dati, columns=["Indice", "Variazione %"])
    print("Estratto file :  variazioni_indici.txt")
    df.to_csv("variazioni_indici.txt", sep='\t', index=False, header=False)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    image_path = "Immagine-2025.jpg"
    ocr_data, _ = run_ocr_with_boxes(image_path)
    estrai_indici_variazioni(ocr_data)
