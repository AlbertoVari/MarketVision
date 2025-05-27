import pytesseract
from gpiozero import LED
import subprocess
import time
import os

# LED
led_verde = LED(17)    # verde → "INTEL"
led_rosso = LED(22)    # rosso → altro testo
led_giallo = LED(27)   # giallo → scatto in corso

def scatta_foto():
    file = "foto.jpg"
    led_giallo.on()
    subprocess.run(["libcamera-still", "-o", file, "-t", "1000", "--nopreview"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)  # LED giallo acceso per 1 secondo
    led_giallo.off()
    return file

try:
    while True:
        path = scatta_foto()

        if os.path.exists(path):
            testo = pytesseract.image_to_string(path, lang="ita").upper().strip()

            if "INTEL" in testo:
                led_verde.on()
                led_rosso.off()
            elif testo:
                led_rosso.on()
                led_verde.off()
            else:
                led_verde.off()
                led_rosso.off()

        time.sleep(5)

except KeyboardInterrupt:
    pass

finally:
    led_verde.off()
    led_rosso.off()
    led_giallo.off()
