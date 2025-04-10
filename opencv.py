import cv2
import numpy as np
import os
import winsound  # For sound alerts on Windows

# Define paths to YOLO files
weights_path = "yolov3.weights"  # Update with your path if necessary
config_path = "yolov3.cfg"        # Update with your path if necessary

# Check if the files exist
if not os.path.isfile(weights_path):
    print(f"Error: Weights file '{weights_path}' not found.")
    exit()

if not os.path.isfile(config_path):
    print(f"Error: Config file '{config_path}' not found.")
    exit()

# Load YOLO
try:
    net = cv2.dnn.readNet(weights_path, config_path)
except cv2.error as e:
    print(f"OpenCV error: {e}")
    exit()
except Exception as e:
    print(f"Error loading YOLO: {e}")
    exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Define crowd density threshold
CROWD_THRESHOLD = 10  # Adjust this value based on your needs

# Gaussian kernel for creating a heatmap effect
def create_heatmap_effect(heatmap, box, sigma=30):
    """Create a Gaussian heatmap effect around the bounding box."""
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2

    # Create a Gaussian distribution centered around the bounding box
    size = 2 * sigma + 1
    x_range = np.arange(0, size)
    y_range = np.arange(0, size)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    gaussian = np.exp(-((x_grid - sigma) * 2 + (y_grid - sigma) * 2) / (2 * sigma ** 2))

    # Apply the Gaussian to the heatmap around the bounding box
    x_start = max(0, center_x - sigma)
    x_end = min(heatmap.shape[1], center_x + sigma + 1)
    y_start = max(0, center_y - sigma)
    y_end = min(heatmap.shape[0], center_y + sigma + 1)

    heatmap[y_start:y_end, x_start:x_end] += gaussian[y_start - (center_y - sigma):y_end - (center_y - sigma), 
                                                         x_start - (center_x - sigma):x_end - (center_x - sigma)]

# Initialize heatmap
heatmap = np.zeros((480, 640), dtype=np.float32)  # Adjust size based on your camera resolution

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    person_count = 0  # Count of detected people

    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for people (class_id 0 for COCO dataset)
            if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if indices are not empty
    if len(indices) > 0:
        for i in indices.flatten():  # Use flatten to handle the indices correctly
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            create_heatmap_effect(heatmap, box)  # Apply heatmap effect to the bounding box
            person_count += 1

    # Decay the heatmap to create a moving effect
    heatmap *= 0.9  # Adjust decay factor as needed
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)

    # Display the results
    cv2.putText(overlay, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if person_count > CROWD_THRESHOLD:
        winsound.Beep(1000, 500)  # Sound alert for crowd density
        cv2.putText(overlay, 'Crowd Alert!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Real-Time Population Detection', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()