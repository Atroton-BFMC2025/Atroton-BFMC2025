from picamera2 import Picamera2

class Camera_init:

    def __init__(self):

          '''self.picam2 = Picamera2()
          self.picam2.preview_configuration.main.size = (320, 180)
          self.picam2.preview_configuration.main.format = "RGB888"
          self.picam2.preview_configuration.align()
          self.picam2.configure("preview")
          self.picam2.start()'''
          self.picam2 = Picamera2()
          config = self.picam2.create_preview_configuration(
          buffer_count=1,
          queue=False,
          main={"format": "RGB888", "size": (320,180)}, # 320 x 180
          lores={"size": (320, 180)},
          encode="lores"
          )
          self.picam2.configure(config)

          self.picam2.start()
        

    def camera_frames(self):
         # Capture and return a frame from the already initialized camera
        return self.picam2.capture_array()

