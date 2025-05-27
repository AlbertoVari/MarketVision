# MarketVision
Edge AI-powered IoT device using Raspberry Pi + Movidius NCS to monitor market index trends via screenshots, OCR, and OpenVINO-based trend detection, triggering LED alerts for traders.

How boring is to see Market display to check index trend ?


My #IoT Edge #AI solve this problem for #traders : a #Raspberry connect to CSI camera take every 5 seconds a screenshot from a display showing trading website such as https://it.investing.com/indices/global-indices and  light on/off led related trend inversion for specific index based on #AI image-to-text model with #NPU accelerator Movidius Neural Network Compute Stick #Intel and #OpenVINO (deploy high-performance #deeplearning inference applications, especially for computer vision tasks, on Intel hardware)


Feature -> Library (Python 3.6) 

Photo capture ->   libcamera-jpeg  

Image reading- > Pillow (PIL)   

OCR -> Tesseract  

DL Inference ->  OpenVINO 2020.3 (MYRIAD device)  

OpenVINO documentation 2020.3 : https://docs.openvino.ai/archives/index.html
Model : text-detection-0004
