import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
from camera_init import Camera_init

# Constants for distance estimation
KNOWN_WIDTH = 11 # Known width of the traffic sign in cm
FOCAL_LENGTH = 300  # Precomputed focal length (calibrated with a reference image)
FRAME_WIDTH = 320  # Camera frame width in pixels (matches Picamera2 config)
FRAME_CENTER_X = FRAME_WIDTH // 2  # Center X of the frame



# Load YOLOv8
model = YOLO("/home/pi/brain_25/Brain/src/algorithms/threads/semifinal_model_3_openvino_model")
# Capture a frame from the camera
camera_init = Camera_init()
while True:
    frame = camera_init.camera_frames()
    
    # Run YOLO model on the captured frame and store the results
    results = model(frame)
    
    # Annotate the frame
    annotated_frame = results[0].plot()
    
    # Process detected objects
    for detection in results[0].boxes:
        conf = detection.conf
        class_idx = int(detection.cls[0])
        if conf >= 0.5:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            detected_width = x2 - x1
            
            # Estimate apparent distance
            apparent_distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / detected_width
            
            # Compute the horizontal offset from the center of the frame
            object_center_x = (x1 + x2) // 2
            offset_x = object_center_x - FRAME_CENTER_X
            
            # Calculate the angle theta using arctan(offset / focal length)
            theta = np.arctan(offset_x / FOCAL_LENGTH)
            
            # Compute the real distance using trigonometry
            real_distance_cm = apparent_distance_cm / np.cos(theta)
            
            # Display distance on frame
            text = f"{real_distance_cm:.1f} cm"
            cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            class_name = model.names[class_idx]
            print("Detected object class:", class_name)

    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    fps_text = f'FPS: {fps:.1f}'
    
    # Draw FPS on frame
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)
    
    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()