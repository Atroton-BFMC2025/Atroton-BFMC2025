import logging
import threading
import time
import numpy as np

from src.utils.messages.allMessages import SteerMotor,mainCamera,Klem,SpeedMotor
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop

class Fsm:
        
    def __init__(self, queuesList, logger, debugger):
        #self.speed = 150
        self.sign_flag = False
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.klSender = messageHandlerSender(self.queuesList,Klem)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        
    def change_speed_pedestrian(self,speed, time_s):
        self.speedMotorSender.send(str(60))
        time.sleep(time_s)
        self.speedMotorSender.send(str(speed))
        self.sign_flag = False
    
    def stop_speed(self,speed,time_s):
        self.speedMotorSender.send(str(0))
        time.sleep(time_s)
        self.speedMotorSender.send(str(speed))
        self.sign_flag = False

    def no_entry_speed(self):
        self.speedMotorSender.send(str(0))

    def parking_action(self, speed):
        self.sign_flag = True

        self.speedMotorSender.send(str(speed))
        time.sleep(3)

        self.speedMotorSender.send(str(0))
        time.sleep(1)

        self.steerMotorSender.send(70)

        self.speedMotorSender.send(str(-speed))
        time.sleep(2)

        self.steerMotorSender.send(-40)
        self.speedMotorSender.send(str(-speed))
        time.sleep(2)

        self.speedMotorSender.send(str(0))
        
        self.sign_flag = False

    def get_action(self,last_seen_label,distance_from_sign):
        
        print("sign flag: ",self.sign_flag )
        

        if last_seen_label == 'pedestrian_sign' and distance_from_sign < 70:
            self.sign_flag = True 
            threading.Thread(target=self.change_speed_pedestrian, args=(120, 3)).start()
        elif last_seen_label == 'parking_sign' and distance_from_sign < 70:
            threading.Thread(target=self.parking_action, args=(120)).start()
            self.sign_flag = True # Parking Not Completed
        elif last_seen_label == 'stop_sign' and distance_from_sign < 70:
            threading.Thread(target=self.stop_speed, args=(120, 5)).start()
            self.sign_flag = True
        elif last_seen_label == 'no_entry_sign' and distance_from_sign < 70:
            threading.Thread(target=self.no_entry_speed).start()
            self.sign_flag = True
        else :
            self.sign_flag == False
            self.speedMotorSender.send(str(150))
     
     
     