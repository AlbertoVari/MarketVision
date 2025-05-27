from openvino.inference_engine import IECore
import cv2
import numpy as np

# Initialize OpenVINO
ie = IECore()

# Load the model
net = ie.read_network(model="face-detection-adas-0001.xml", weights="face-detection-adas-0001.bin")
exec_net = ie.load_network(network=net, device_name="CPU")

# Get input/output names
input_blob = next(iter(net.inputs))
output_blob = next(iter(net.outputs))

# Read and preprocess image
image = cv2.imread("your_image.jpg")
resized = cv2.resize(image, (672, 384))         # Resize to model input
input_image = resized.transpose((2, 0, 1))       # Convert HWC to CHW
input_image = np.expand_dims(input_image, axis=0)

# Inference
result = exec_net.infer(inputs={input_blob: input_image})

# Draw results
for detection in result[output_blob][0][0]:
    _, _, conf, x_min, y_min, x_max, y_max = detection
    if conf > 0.5:
        xmin = int(x_min * image.shape[1])
        ymin = int(y_min * image.shape[0])
        xmax = int(x_max * image.shape[1])
        ymax = int(y_max * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
