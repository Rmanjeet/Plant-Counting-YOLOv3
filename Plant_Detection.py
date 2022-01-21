import cv2
import numpy as np
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet('yolo_model.weights', 'yolo_model.cfg')
classes = ['cabbage']
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")
cap = cv2.VideoCapture('cabbage.mp4')

if (cap.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size)
count = []
while True:
  ret, img = cap.read()
  if ret == False:
    break
  height, width, channels = img.shape
  #roi = img[80:400,154:700]
  blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

  # Detecting objects
  net.setInput(blob)
  outs = net.forward(output_layers)

  # Showing informations on the screen
  class_ids = []
  confidences = []
  boxes = []
  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence >0.5:
        # Object detected
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  # We use NMS function in opencv to perform Non-maximum Suppression
  # we give it score threshold and nms threshold as arguments.
  #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  box_ids=[]
  for box in boxes:
    if ((2*box[0])+box[2])//2 > 154 and ((2*box[0])+box[2])//2 <700:
      if ((2*box[1])+box[3])//2 > 80 and ((2*box[1])+box[3])//2 <400:
        box_ids.append(box)
  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  cv2.rectangle(img, (154,80),(700,400), (0,0,255), 2)
  box_ids = tracker.update(box_ids)
  for i in range(len(box_ids)):
    if i in indexes:
        x, y, w, h, id = box_ids[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        if id in count:
          pass
        else:
          count.append(id)
  cv2.putText(img, 'Plant Counter : '+str(len(count)), (50,50),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2)
  result.write(img)
  #cv2.imshow('Img', img)
  if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
