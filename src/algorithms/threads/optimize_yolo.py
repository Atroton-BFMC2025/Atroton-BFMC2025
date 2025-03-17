from ultralytics import YOLO

model = YOLO('/home/pi/brain_25/Brain/src/algorithms/threads/semifinal_model_3_openvino_model')
result = model('/home/pi/brain_25/Brain/src/algorithms/threads/shopping.jpeg')
#vivomodel=model.export(format='openvivo')