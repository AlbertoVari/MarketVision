import pytesseract
import subprocess
import time
import os
import re
import cv2
import numpy as np
from gpiozero import LED
from openvino.inference_engine import IECore

# === OpenVINO model paths ===
model_xml = "models/intel/text-detection-0004/FP16/text-detection-0004.xml"
model_bin = "models/intel/text-detection-0004/FP16/text-detection-0004.bin"

# === GPIO LEDs ===
led_green = LED(17)   # Green
led_red = LED(22)     # Red
led_yellow = LED(27)  # Yellow

# === Load OpenVINO model ===
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="MYRIAD")

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape

previous_value = None

def capture_photo():
    filename = "snapshot.jpg"
    led_yellow.on()
    subprocess.run(["libcamera-still", "-o", filename, "-t", "1000", "--nopreview"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    led_yellow.off()
    return filename

def detect_text_boxes(image):
    resized = cv2.resize(image, (w, h))
    input_image = resized.transpose((2, 0, 1))[np.newaxis, :]
    result = exec_net.infer({input_blob: input_image})
    boxes = result[output_blob][0][0]
    return boxes

def extract_sp500_change(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    boxes = detect_text_boxes(image)
    orig = image.copy()
    candidate_lines = []

    for box in boxes:
        _, _, conf, x_min, y_min, x_max, y_max = box
        if conf > 0.5:
            xmin = int(x_min * orig.shape[1])
            ymin = int(y_min * orig.shape[0])
            xmax = int(x_max * orig.shape[1])
            ymax = int(y_max * orig.shape[0])
            roi = orig[ymin:ymax, xmin:xmax]
            text = pytesseract.image_to_string(roi, config="--psm 6").strip().upper()
            candidate_lines.append((text, xmin, ymin))

    # Try to find line containing S&P 500 and its associated Chg. %
    for i, (line, x, y) in enumerate(candidate_lines):
        if "S&P 500" in line:
            for j in range(i, min(i+3, len(candidate_lines))):
                match = re.search(r"([-+]?\d{1,3},\d{1,2})%", candidate_lines[j][0])
                if match:
                    return float(match.group(1).replace(",", "."))
    return None

def update_leds(current, previous):
    if current is None:
        print("⚠️ Couldn't extract value.")
        led_green.off()
        led_red.off()
        led_yellow.off()
        return

    if previous is None:
        print(f"ℹ️ First reading: {current}%")
        return
 
    if current == previous:
        print("⏸️ No Change → Yellow LED")
        led_yellow.on(); led_green.off(); led_red.off()
    elif (current > previous and current >= 0) or (current > previous and current < 0 and previous < 0):
        print("📈 Improved → Green LED")
        led_green.on(); led_red.off(); led_yellow.off()
    else:
        print("📉 Worsened → Red LED")
        led_red.on(); led_green.off(); led_yellow.off()


try:
    while True:
        photo = capture_photo()
        current_value = extract_sp500_change(photo)
        print(f"S&P 500 Chg. %: {current_value}")
        update_leds(current_value, previous_value)
        previous_value = current_value
        time.sleep(5)

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")

finally:
    led_green.off()
    led_red.off()
    led_yellow.off()
