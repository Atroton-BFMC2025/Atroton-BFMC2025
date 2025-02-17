
import threading
import base64
import cv2 as cv 
import cv2 
import numpy as np
from time import time
import logging
import math
import matplotlib as plt 
from picamera2 import Picamera2

from src.utils.messages.allMessages import SteerMotor,mainCamera,Klem,SpeedMotor
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.algorithms.threads.lane_keeping import *
from ultralytics import YOLO



class main_cv_Thread(ThreadWithStop):
    """Thread which will handle lane keeping functionalities.\n
    Args:
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(main_cv_Thread, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 5
        # define a range of black color in HSV
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([227, 100, 70])
        # Rectangular Kernel
        self.rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        self.klSender = messageHandlerSender(self.queuesList,Klem)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.speedMotorSender =messageHandlerSender(self.queuesList, SpeedMotor)
        self.cameraSender = messageHandlerSender(self.queuesList, mainCamera)

        self.subscribe()

    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList,mainCamera,"lastOnly",True)
        
    # =============================== STOP ================================================
    def stop(self):
        super(main_cv_Thread, self).stop()

    # ================================ RUN ================================================

    
    def run(self):
        print('Started')
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            buffer_count=1,
            queue=False,
            main={"format": "RGB888", "size": (320,180)},
            lores={"size": (320, 180)},
            encode="lores"
        )
        picam2.configure(config)
        #picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)}) 
        picam2.start()
        self.klSender.send("30")
        model = YOLO("/home/pi/brain_25/yolo/semifinal_model_3.torchscript")

        
   
        
        #cap = cv.VideoCapture("bosch_test.mp4")

    
        
        while self._running:
            try:
                lane_follower =LaneFollower()
                frame = picam2.capture_array()


                #_, frame = cap.read()

                # apply some gaussian blur to the image
                kernel_size = (3, 3)
                gauss_image = cv.GaussianBlur(frame, kernel_size, 0)
                # gauss_image =  cv.bilateralFilter(frame, 9, 75, 75)

                # here we convert to the HSV colorspace
                hsv_image = cv.cvtColor(gauss_image, cv.COLOR_BGR2HSV)

                # apply color threshold to the HSV image to get only black colors
                thres_1 = cv.inRange(hsv_image, self.lower_black, self.upper_black)

                # dilate the threshold image
                thresh = cv.dilate(thres_1, self.rectKernel, iterations=1)
                
                # apply canny edge detection
                low_threshold = 200
                high_threshold = 400
                canny_edges = cv.Canny(gauss_image, low_threshold, high_threshold) 

                roi_image = region_of_interest(canny_edges)

                line_segments = detect_line_segments(roi_image)
                lane_lines = average_slope_intercept(frame, line_segments)
                steering=lane_follower.get_steering_angle(frame,lane_lines)
                normalized = (steering-90)*12
                #normalized_new = 1.5 * normalized
                self.steerMotorSender.send(str(normalized))
                if normalized==0 :
                    self.speedMotorSender.send(str(200))
                else :
                    #self.speedMotorSender.send(str(int(200 - 65*(90/abs(normalized)))))
                    self.speedMotorSender.send(str(120))
    
                heading_image1 =lane_follower.display_heading_line( frame, steering) 
                
                
                #cv.imshow('raw', frame )
                cv.imshow('heading', heading_image1)
                line_image = display_lines(frame, lane_lines)
                #cv.imshow('Line Image', line_image)
                #cv.imshow('canny',canny_edges)
                #cv.imshow('roi',roi_image)
                
                #SIGNS DETECTION - YOLO8N
                results = model(frame)
                
    
                # Output the visual detection data, we will draw this on our camera preview window
                annotated_frame = results[0].plot()
                
                # Get inference time
                inference_time = results[0].speed['inference']
                fps = 1000 / inference_time  # Convert to milliseconds
                text = f'FPS: {fps:.1f}'

                # Define font and position
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
                text_y = text_size[1] + 10  # 10 pixels from the top

                # Draw the text on the annotated frame
                cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display the resulting frame
                cv2.imshow("Camera", annotated_frame)

    
        
                print('steering :',normalized)
                keyboard = cv.waitKey(30)
                if keyboard == ord('q') or keyboard == 27:
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during processing: {e}")
               # break if laneKeeping:
                
        
        print('Stopped')
        # =============================== START ===============================================
    def start(self):
        super(main_cv_Thread, self).start()


    
