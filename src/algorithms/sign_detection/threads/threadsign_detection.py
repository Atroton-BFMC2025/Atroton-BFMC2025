import threading
import base64
import cv2 as cv 
import cv2 
import numpy as np
from time import time
import logging
import math
import matplotlib as plt 
from ultralytics import YOLO
#from picamera2 import Picamera2

#from src.algorithms.threads.camera_init import Camera_init
from src.algorithms.fsm import Fsm

from src.utils.messages.allMessages import SteerMotor,mainCamera,Klem,SpeedMotor
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
class threadsign_detection(ThreadWithStop):
    """This thread handles sign_detection.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.speed = 150
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.subscribe()
        super(threadsign_detection, self).__init__()
        #self.camera_init = Camera_init()
        self.cameraSender = messageHandlerSender(self.queuesList, mainCamera)
        self.speedMotorSender =messageHandlerSender(self.queuesList, SpeedMotor)
        self.klSender = messageHandlerSender(self.queuesList,Klem)
        self.speedMotorSender.send(str(self.speed))  # Add this line
    def subscribe(self):
        """Subscribes to the messages you are interested in"""
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList,mainCamera,"lastOnly",True)
    pass


    def run(self):
        # Constants for distance estimation
        KNOWN_WIDTH = 5# Known width of the traffic sign in cm
        FOCAL_LENGTH = 275  # Precomputed focal length (calibrated with a reference image)
        FRAME_WIDTH = 256  # Camera frame width in pixels (matches Picamera2 config)
        FRAME_CENTER_X = FRAME_WIDTH // 2  # Center X of the frame
        car_action = Fsm(self.queuesList, self.logging, self.debugging)
        # Load YOLOv8
        model = YOLO("/home/pi/brain_25/Brain/src/algorithms/threads/semifinal_model_3_openvino_model")
        # Capture a frame from the camera
        while self._running:
            #frame = self.camera_init.camera_frames()
            frame_data = self.serialCameraSubscriber.receive()
            if frame_data is None:
                continue  # No new frame available

            # Decode the base64 image data
            frame_bytes = base64.b64decode(frame_data)
            frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode image")
                continue
            # Run YOLO model on the captured frame and store the results
            results = model(frame)
            
            # Annotate the frame
            annotated_frame = results[0].plot()
            
            # Process detected objects
            for detection in results[0].boxes:
                conf = detection.conf
                class_idx = int(detection.cls[0])
                if conf >= 0.4:
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
                    #text = f"{real_distance_cm:.1f} cm"
                    #cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    class_name = model.names[class_idx]
                    print("Detected object class:", class_name)
                    print("dist from {} is {}".format(class_name, real_distance_cm))
                    #if class_name == 'stop_sign' and real_distance_cm < 100:
                        #self.speedMotorSender.send(str(0))
                        #self.klSender.send("30")
                    # Get inference time
                    inference_time = results[0].speed['inference']
                    fps = 1000 / inference_time  # Convert to milliseconds
                    fps_text = f'FPS: {fps:.1f}'
                    
                    # Draw FPS on frame
                    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display the resulting frame
                    #cv2.imshow("Camera", annotated_frame)

                    car_action.get_action(class_name,real_distance_cm)
                    # Exit the program if q is pressed
            if cv2.waitKey(1) == ord("q"):
                break